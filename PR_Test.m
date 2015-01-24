clc;
clear;
delfigs;
prwaitbar off;
prwarning off

nist_data = prnist(0:9,1:1000)
% NIST EVAL
prmemory(64000000);
clc;

iter = 10;        % Number of performance evaluations
num_test = 10;  % Number of test objects per class
avarage = 0;
classify = ldc;
parametric = {fisherc, ldc,loglc,nnm,nmsc,quadrc,qdc,udc,klldc,pcldc,polyc,subsc,knnc,parzenc,parzendc,naivebc,bpxnc,perlc,svc};
averageTotal = zeros(size(parametric));
averageTime = zeros(size(parametric));

for j = 1:size(parametric,2)
    classify = parametric{j};
    tic
    avarage = 0;
    for i = 1:iter
        % Generate a random training set with 10 objects per class 

        trainingSize = 0.01;
        [train, test] = gendat(nist_data, trainingSize);
        % Calculate trainings prdataset object

        trn_unselected = my_rep1(train);

        [mapping, R] = pcam(trn_unselected,24);
        trn_featsel = trn_unselected*mapping;

        classifier = classify(trn_featsel);

        mapped_classifier = mapping*classifier;

        e = nist_eval('my_rep1', mapped_classifier, num_test);
        average = average + e;

    end
    averageTime(j) = toc/iter;
    averageTotal(j) = average/iter;
    average = average/iter
    average
end

averageTotal
averageTime
