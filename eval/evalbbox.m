function evalRes=evalbbox(sel_box,gtarray)
%     fprintf('Visbbox for %s \n',nm);
    thr_num = [1 2 5 10 20 50 100 200 500 1000 2000];
    thr_IoU = [0.5 0.7];
    max_num = size(sel_box,1);
    thr_num(thr_num>max_num)=max_num;
    N = size(thr_num,2);
    M = size(thr_IoU,2);
    evalRes = zeros(M,N);
    
    for i = 1:N
        for j = 1:M
            [~,~,objIoU]=scorebboxes(gtarray,sel_box(1:thr_num(i),:),thr_IoU(j));
            objmaxIoU = max(objIoU,[],1);
            hit_obj = objmaxIoU >= thr_IoU(j);
            evalRes(j,i)=sum(hit_obj);
        end
    end
    
end
