clc;
clear;
delfigs;
prwaitbar off;
prwarning off;

nist_data = prnist(0:9,1:1000)
% NIST EVAL
prmemory(128000000);
clc;

iter = 1;        % Number of performance evaluations
num_test = 100;  % Number of test objects per class
average = 0;
parametric = {fisherc, ldc,loglc,nmc,nmsc,quadrc,qdc,udc,klldc,pcldc,polyc,subsc,knnc,parzenc,parzendc,naivebc,bpxnc,perlc,vpc};
averageNist = zeros(size(parametric));
averageTime = zeros(size(parametric));
ClassEval = [];

for j = 1:size(parametric,2)
    classify = parametric{j};
    tic
    average = 0;
    for i = 1:iter
        % Generate a random training set with 10 objects per class 
        clc
        fprintf('Classifier %i of %i\n',j,size(parametric,2));
        fprintf('Iteration: %i of %i\n',i,iter);
        trainingSize = 0.01;
        testSize = trainingSize/0.1;
        [train, ~] = gendat(nist_data, trainingSize);
        [test, ~] = gendat(nist_data, testSize);
        % Calculate trainings prdataset object

        trn_unselected = my_rep1(train);
        test_unselected = my_rep1(test);

        [mapping,~] = pcam(trn_unselected,24);
        
        trn_featsel = trn_unselected*mapping;
        test_featsel = test_unselected*mapping;

        W = classify(trn_featsel);

        mapped_classifier = mapping*W;
        e_n = nist_eval('my_rep1', mapped_classifier, num_test);
        average = average + e_n;
    end
    averageTime(j) = toc/iter;
    ClassEval = [ClassEval;clevalf(trn_featsel,classify,24,[],1,test_featsel)];
    averageNist(j) = average/iter;
end
fprintf('Results:\n');
for j = 1:size(parametric,2)
    disp(ClassEval(j));
    fprintf('Average nist_eval: %f - %f sec\n',averageNist(j),averageTime(j));
end
