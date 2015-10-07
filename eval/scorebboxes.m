function [label,overlap,objIoU]=scorebboxes(gtarray,bboxes,PosWinThr)
% gtarray: nx4, n: # of objects, each row is [x1,y1,x2,y2]
% bboxes: Nx4, n: # of bboxes in this particular image, each row is [x1,y1,x2,y2]
% label: Nx1, label of each bbox
% overlap: Nx1, maximum overlap of each bbox with ground truth
gt_area = (gtarray(:,3)-gtarray(:,1)).*(gtarray(:,4)-gtarray(:,2));
bb_area = (bboxes(:,3)-bboxes(:,1)).*(bboxes(:,4)-bboxes(:,2));

%Find the zeros in gt area 
zero_idx = (gt_area == 0);
if any(zero_idx)
    if ((gtarray(zero_idx,3)-gtarray(zero_idx,1)+1)==0)
        gt_area(zero_idx) = (gtarray(zero_idx,3)-gtarray(zero_idx,1)+0.01).*(gtarray(zero_idx,4)-gtarray(zero_idx,2));
        gt_wh = cat(2,gtarray(:,1:2),gtarray(:,3)-gtarray(:,1)+0.01,gtarray(:,4)-gtarray(:,2));
    else
        gt_area(zero_idx) = (gtarray(zero_idx,3)-gtarray(zero_idx,1)).*(gtarray(zero_idx,4)-gtarray(zero_idx,2)+0.01);
        gt_wh = cat(2,gtarray(:,1:2),gtarray(:,3)-gtarray(:,1),gtarray(:,4)-gtarray(:,2)+0.01);
    end
else
    gt_wh = cat(2,gtarray(:,1:2),gtarray(:,3)-gtarray(:,1),gtarray(:,4)-gtarray(:,2));
end

bb_wh = cat(2,bboxes(:,1:2),bboxes(:,3)-bboxes(:,1),bboxes(:,4)-bboxes(:,2));
Int_area = rectint(bb_wh,gt_wh);

Union_area = repmat(bb_area,1,size(gt_area,1))+repmat(gt_area',size(bb_area,1),1);
objIoU = Int_area./(Union_area-Int_area);

%Label sliding window by maximum overlap with objects
[overlap,~] = max(objIoU,[],2);
overlap = floor(overlap*1000)/1000;
PosWinIdx = overlap>=PosWinThr;
label = zeros(size(bboxes,1),1);
label(PosWinIdx) = 1;


end    