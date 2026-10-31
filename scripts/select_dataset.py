"""Select a deterministic, visually diverse Indoor67 working set; prune only with --prune."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from scenelens import config

CATEGORIES = ['bathroom','bedroom','classroom','dining_room','kitchen',
              'library','livingroom','meeting_room','office','restaurant']
SEED = 20261031


def describe(path):
    image=cv2.imread(str(path))
    if image is None or min(image.shape[:2]) < 120:
        return None
    h,w=image.shape[:2]
    thumb=cv2.resize(image,(160,120),interpolation=cv2.INTER_AREA)
    gray=cv2.cvtColor(thumb,cv2.COLOR_BGR2GRAY)
    hsv=cv2.cvtColor(thumb,cv2.COLOR_BGR2HSV)
    hist=cv2.calcHist([hsv],[0,1],None,[12,4],[0,180,0,256]).reshape(-1)
    hist=np.sqrt(hist/hist.sum())
    small=cv2.resize(gray,(9,8),interpolation=cv2.INTER_AREA)
    dhash=sum(int(bit)<<i for i,bit in enumerate((small[:,1:]>small[:,:-1]).reshape(-1)))
    sharpness=float(cv2.Laplacian(gray,cv2.CV_32F).var())
    stats=np.array([gray.mean()/255,gray.std()/128,np.log1p(sharpness)/10,
                    np.log(w/h),np.log(w*h)/15],dtype=np.float32)
    return dict(path=path,width=w,height=h,brightness=float(gray.mean()),
                contrast=float(gray.std()),sharpness=sharpness,dhash=dhash,
                feature=np.concatenate([hist,stats]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prune',action='store_true',help='Delete all unselected source images after validation.')
    args=parser.parse_args()
    cv2.setNumThreads(2)
    root=config.DATASET_IMAGES_DIR
    manifest=config.DATASET_SPLITS_DIR/'selection.json'
    if manifest.exists():
        report=json.loads(manifest.read_text())
        print('Using existing selection manifest; no reselection.')
    else:
        originals={name:set((config.DATASET_SPLITS_DIR/f'{name}.txt').read_text().splitlines())
                   for name in ['train','test']}
        selected=[]
        seen_hashes=set()
        for category in CATEGORIES:
            candidates=[]
            for path in sorted((root/category).iterdir()):
                if path.suffix.lower() not in {'.jpg','.jpeg','.png'}: continue
                item=describe(path)
                if item is not None: candidates.append(item)
            if len(candidates)<20: raise RuntimeError(f'Not enough readable images: {category}')
            features=np.stack([item['feature'] for item in candidates])
            # Normalize scalar descriptors while retaining the color histogram's distances.
            scalars=features[:,-5:]
            low,high=np.percentile(scalars,[5,95],axis=0)
            features[:,-5:]=np.clip((scalars-low)/np.maximum(high-low,.01),0,1)*.6
            center=np.median(features,axis=0)
            order=list(np.argsort(np.square(features-center).sum(axis=1)))
            chosen=[]
            while len(chosen)<20:
                index=int(order.pop(0))
                item=candidates[index]
                if any((item['dhash']^candidates[j]['dhash']).bit_count()<8 for j in chosen):
                    if not order: raise RuntimeError(f'Near-duplicate filtering exhausted {category}')
                    continue
                digest=hashlib.sha256(item['path'].read_bytes()).hexdigest()
                if digest in seen_hashes: continue
                chosen.append(index)
                seen_hashes.add(digest)
                distances=np.min(np.square(features[:,None,:]-features[chosen][None,:,:]).sum(axis=2),axis=1)
                order=[int(i) for i in np.argsort(-distances) if i not in chosen]
            # Fixed regression examples: typical, low-light, and high-detail among retained images.
            regression=[chosen[0]]
            for key in ['brightness','sharpness']:
                ranked=sorted(chosen,key=lambda i:candidates[i][key],reverse=key=='sharpness')
                regression.append(next(i for i in ranked if i not in regression))
            for index in chosen:
                item=candidates[index]
                relative=item['path'].relative_to(root).as_posix()
                selected.append(dict(path=relative,category=category,
                    subset='regression' if index in regression else 'development',
                    width=item['width'],height=item['height'],brightness=item['brightness'],
                    contrast=item['contrast'],sharpness=item['sharpness'],
                    sha256=hashlib.sha256(item['path'].read_bytes()).hexdigest(),
                    original_split=next((k for k,v in originals.items() if relative in v),'unassigned')))
            print(f'{category}: selected 20 from {len(candidates)} readable candidates',flush=True)
        report=dict(seed=SEED,method='Deterministic max-min color/brightness/contrast/sharpness/aspect/resolution diversity; within-category dHash distance >=8; unique SHA256.',
                    categories=CATEGORIES,count=200,development=170,regression=30,images=selected)
        manifest.write_text(json.dumps(report,indent=2)+'\n')
    records=report['images']
    keep={item['path'] for item in records}
    assert len(keep)==200
    assert Counter(item['category'] for item in records)=={category:20 for category in CATEGORIES}
    assert Counter(item['subset'] for item in records)=={'development':170,'regression':30}
    for item in records:
        path=root/item['path']
        assert path.is_file() and cv2.imread(str(path)) is not None,path
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'],path
    for subset in ['development','regression']:
        paths=sorted(item['path'] for item in records if item['subset']==subset)
        (config.DATASET_SPLITS_DIR/f'{subset}.txt').write_text('\n'.join(paths)+'\n')
    preview=config.PROJECT_ROOT/'data/outputs/dataset-selection'
    preview.mkdir(parents=True,exist_ok=True)
    for category in CATEGORIES:
        canvas=np.full((4*180,5*240,3),245,np.uint8)
        items=[item for item in records if item['category']==category]
        for index,item in enumerate(items):
            image=cv2.imread(str(root/item['path']))
            h,w=image.shape[:2]
            scale=min(230/w,145/h)
            image=cv2.resize(image,(max(1,round(w*scale)),max(1,round(h*scale))))
            row,col=divmod(index,5)
            top,left=row*180+(145-image.shape[0])//2,col*240+(240-image.shape[1])//2
            canvas[top:top+image.shape[0],left:left+image.shape[1]]=image
            text=f'{index+1:02d} {Path(item["path"]).name[:20]}'
            cv2.putText(canvas,text,(col*240+5,row*180+160),cv2.FONT_HERSHEY_SIMPLEX,.36,(0,0,0),1,cv2.LINE_AA)
            if item['subset']=='regression':
                cv2.putText(canvas,'REGRESSION',(col*240+5,row*180+175),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,100,0),1)
        cv2.imwrite(str(preview/f'{category}.jpg'),canvas)
    if args.prune:
        removed=0
        freed=0
        for path in list(root.rglob('*')):
            if path.is_file() and path.relative_to(root).as_posix() not in keep:
                freed+=path.stat().st_size
                path.unlink()
                removed+=1
        for path in sorted(root.rglob('*'),key=lambda p:len(p.parts),reverse=True):
            if path.is_dir() and not any(path.iterdir()):path.rmdir()
        for name in ['train.txt','test.txt']:
            (config.DATASET_SPLITS_DIR/name).unlink(missing_ok=True)
        assert {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}==keep
        print(f'Retained 200 images; removed {removed}; freed {freed/2**30:.2f} GiB.')
    print(f'Manifest: {manifest}; contact sheets: {preview}')


if __name__=='__main__':
    main()
