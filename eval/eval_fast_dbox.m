% Setup directories
function eval_fast_dbox()
name = 'fast-dbox-multiscale';
suffix = ['results_' name];
addpath('./MSCOCO/MatlabAPI');
dataDir='./MSCOCO'; 
split = 'val';
year = '2014';
dataType = [split year];
annFile=sprintf('%s/annotations/instances_%s.json',dataDir,dataType);

%% load coco
coco=CocoApi(annFile);
imgIds = coco.getImgIds();
num_imgs = numel(imgIds);

fprintf('loading gtbbox ...\n');
load(['./data/coco_matlab_data/COCO_' split '_gtbbox.mat']);
fprintf('loading fast dboxes score...\n');
load(['./output/default/coco_val2014/' name '/fast_dbox_output_scores.mat']);

fprintf('loading edge boxes ...\n');
load(['./data/edge_box_data/' split '2014.mat']);
eboxes = boxes;
clear('boxes');


mean_dbox_Mprop  = zeros(2,11);
mean_edgebox = zeros(2,11);
sum_gt = 0;

%proxy vector
num_iter = numel(eboxes);
for img_id = 1:num_iter
    fprintf('Evaluate fast DeepBox COCO %s images:%d\n',split,img_id);
    sel_gtarray = gtbbox{img_id};
    sel_num_objs = size(sel_gtarray,1);
    [~,I_sort] = sort(score_list{img_id}(:,2),'descend');
    trim_bbs_m = eboxes{img_id}(I_sort,:);
    boxes = eboxes{img_id};
    
    %Evaluation of boxes
    if isempty(sel_gtarray)
        fprintf('img %d no specified categories .... \n',img_id);
    else
        evalRes_dbox_Mprop =evalbbox(trim_bbs_m,sel_gtarray);
        evalRes_edgebox=evalbbox(boxes,sel_gtarray);
        mean_dbox_Mprop  = mean_dbox_Mprop +evalRes_dbox_Mprop;
        mean_edgebox = mean_edgebox+evalRes_edgebox;
        sum_gt = sum_gt + size(sel_gtarray,1);
    end
    
    
end

fprintf('Fast DeepBox evaluation results:\n');
display(mean_dbox_Mprop/sum_gt);
fprintf('Edge boxes evaluation results:\n');
display(mean_edgebox/sum_gt);
fprintf('Difference:\n');
display((mean_dbox_Mprop-mean_edgebox)/sum_gt);

fprintf('Save to file ...');

save(['./eval/' suffix '.mat'],'mean_dbox_Mprop','mean_edgebox','sum_gt','-v7');
fprintf('done\n');

fprintf('Plot results and compute AUC....\n');
plot_fast_rcnn(name);
