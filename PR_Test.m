clc;
clear;
delfigs;
prwaitbar off;

nist_data = prnist(0:9,1:1000)
% NIST EVAL
prmemory(64000000);
clc;

iter = 1;        % Number of performance evaluations
num_test = 100;  % Number of test objects per class
avarage = 0;

for i = 1:iter
    % Generate a random training set with 10 objects per class 
    trainingSize = 0.4;
    [train, test] = gendat(nist_data, trainingSize);
    % Calculate trainings prdataset object

    trn_unselected = my_rep1(train);

    [mapping, R] = pcam(trn_unselected,24);
    show(mapping)
    trn_featsel = trn_unselected*mapping;
    % Train SVC classifier
    %w_fisher = fisherc(trn_featsel);
    t = parzenc;
    w_svc = t(trn_featsel);
    
    %w_fisher_map = mapping*w_fisher;
    w_svc_map = mapping*w_svc;
    
    %E1 = nist_eval('my_rep2', w_fisher_map, num_test);
    e = nist_eval('my_rep1', w_svc_map, num_test)
    avarage = avarage + e;
    %errors(i,1) = E1;   % Fisher
    %errors(i,1) = E2;   % SVC
end
avarage = avarage/iter;
avarage

