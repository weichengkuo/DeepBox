function plot_fast_rcnn(name)
load(['fast_rcnn_dbox_' name '.mat']);
dr_e = mean_edgebox./sum_gt;
dr_ds = mean_dbox_Mprop./sum_gt; 
thr_num = [1 2 5 10 20 50 100 200 500 1000 2000];
auc_thr = log10(thr_num)/log10(2000);
AUC_edge = zeros(1,2);
AUC_dbox = zeros(1,2);
for i = 1:2
    xv = cat(2,0,dr_e(i,:),0);
    yv = cat(2,0,auc_thr,1);
    AUC_edge(i) = polyarea(xv,yv);
    xds = cat(2,0,dr_ds(i,:),0);
    yds = cat(2,0,auc_thr,1);
    AUC_ds(i) = polyarea(xds,yds);
end
figure(1);
semilogx(thr_num,dr_e(1,:),'linewidth',3,'color','b');hold on;
semilogx(thr_num,dr_ds(1,:),'linewidth',3,'color','r');
axh = gca;
set(axh,'XTick',thr_num,'Fontsize',14);
h = legend('Edgebox','DeepBox FastRCNN');
set(h,'Location','NorthWest');
title('COCO Evaluation IoU=0.5','Fontsize',14,'Fontweight','bold');
xlabel('Number of proposals','Fontsize',14,'Fontweight','bold');
ylabel('Recall','Fontsize',14,'Fontweight','demi');
axis([1 2000 0 0.80]);
hold off;


figure(2);
semilogx(thr_num,dr_e(2,:),'linewidth',3,'color','b');hold on;
semilogx(thr_num,dr_ds(2,:),'linewidth',3,'color','r');
axh = gca;
set(axh,'XTick',thr_num,'Fontsize',14);
h = legend('Edgebox','DeepBox FastRCNN');
set(h,'Location','NorthWest');
title('COCO Evaluation IoU=0.7','Fontsize',14,'Fontweight','bold');
xlabel('Number of proposals','Fontsize',14,'Fontweight','bold');
ylabel('Recall','Fontsize',14,'Fontweight','demi');
axis([1 2000 0 0.6]);
hold off;

fprintf('DeepBox FRCNN AUC:\n');
display(AUC_ds);
fprintf('EdgeBox AUC:\n');
display(AUC_edge);
fprintf('Ratio FRCNN AUC to EdgeBox AUC:\n');
display(AUC_ds./AUC_edge);
    
