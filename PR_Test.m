clc;
clear;
delfigs;
prwaitbar off;

nist_data = prnist(0:9,1:1000)
% NIST EVAL
prmemory(64000000);
clc;

iter = 10;        % Number of performance evaluations
num_test = 10;  % Number of test objects per class
avarage = 0;
classify = parzenc;

for i = 1:iter
    % Generate a random training set with 10 objects per class 
    trainingSize = 0.05;
    [train, test] = gendat(nist_data, trainingSize);
    % Calculate trainings prdataset object

    trn_unselected = my_rep1(train);

    [mapping, R] = pcam(trn_unselected,24);
    show(mapping)
    trn_featsel = trn_unselected*mapping;
    % Train SVC classifier
    %w_fisher = fisherc(trn_featsel);
    
    classifier = classify(trn_featsel);
    
    %w_fisher_map = mapping*w_fisher;
    mapped_classifier = mapping*classifier;
    
    %E1 = nist_eval('my_rep2', w_fisher_map, num_test);
    e = nist_eval('my_rep1', mapped_classifier, num_test)
    avarage = avarage + e;
end
avarage = avarage/iter;
avarage

