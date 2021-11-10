function [SVMstruct,svIndex,pe_tr,pe_te]=SVM_clas(X1,y1,X2,y2,tol,C,sigma)
    options = svmsmoset('TolKKT',tol,'Display','iter','MaxIter',20000,'KernelCacheLimit',10000);
    [SVMstruct,svIndex]=svmtrain(X1', y1','KERNEL_FUNCTION','rbf','RBF_SIGMA',sigma,'BOXCONSTRAINT',C,'showplot',true,'Method','SMO','SMO_Opts',options);
    train_res=svmclassify(SVMstruct,X1');
    pe_tr=sum(y1'~=train_res)/length(y1);
    test_res=svmclassify(SVMstruct,X2');
    pe_te=sum(y2'~=test_res)/length(y2);
%     fitcsvm(X1', Y','DeltaGradientTolerance', )