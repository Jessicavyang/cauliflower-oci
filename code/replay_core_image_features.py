"""Replay the three OCI image inputs with the original v8 functions.

This is same-image computational verification, not segmentation validation.
No resizing, color correction, physical scaling or algorithm tuning is added.
"""
from pathlib import Path
import argparse,csv,hashlib,json,sys,importlib.util
BASE=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--limit',type=int,default=3);p.add_argument('--output-dir',type=Path,default=BASE/'outputs/image_replay');a=p.parse_args()
    if a.limit<1 or a.limit>30:p.error('--limit must be between 1 and the 30 saved QA images')
    import cv2 as cv
    import numpy as np
    spec=importlib.util.spec_from_file_location('oci_original_v8',BASE/'code/image_processing/opencv_v8_original.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with (BASE/'data/image_manifests/QA30_image_byte_linkage.csv').open(encoding='utf-8-sig',newline='') as f:rr=list(csv.DictReader(f))[:a.limit]
    output=[]
    for r in rr:
        f=BASE/r['package_image_path'];assert f.is_file(),f'Extract F2 image archive first: {f.name}'
        assert hashlib.sha256(f.read_bytes()).hexdigest()==r['sha256']
        img=cv.imread(str(f));assert img is not None
        gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY);blur=cv.GaussianBlur(gray,(5,5),0);threshold,binary=cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        contours,_=cv.findContours(binary,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE);assert contours
        contour=max(contours,key=cv.contourArea);mask=np.zeros_like(gray);cv.drawContours(mask,[contour],-1,255,-1)
        gaps,_=module.analyze_gaps_internal(mask,binary);hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV);s=float(cv.meanStdDev(hsv,mask=mask)[0][1][0])
        count=int(gaps['num_gaps']);ratio=float(gaps['gap_ratio']);dc=count-float(r['expected_Struct_num_gaps']);dr=ratio-float(r['expected_Struct_gap_ratio']);ds=s-float(r['expected_Color_S'])
        assert dc==0 and abs(dr)<1e-8 and abs(ds)<1e-7,(r['qa_image_id'],dc,dr,ds)
        output.append(dict(qa_image_id=r['qa_image_id'],package_image_path=r['package_image_path'],Otsu_threshold=float(threshold),Struct_num_gaps=count,Struct_gap_ratio=ratio,Color_S=s,delta_count=dc,delta_ratio=dr,delta_Color_S=ds))
    a.output_dir.mkdir(parents=True,exist_ok=True)
    result=dict(status='PASS',n_images=len(output),scope='Same byte-identical images and three core features only; not independent segmentation or acquisition accuracy.',opencv=cv.__version__,numpy=np.__version__,results=output)
    (a.output_dir/'image_replay_result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
