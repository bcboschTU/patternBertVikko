clc;
clear;
delfigs;
prwaitbar off;
prwarning off

%create data set
nist_data = prnist(0:9,1:1000)
prmemory(64000000);
clc;

iter = 10;        % Number of performance evaluations
num_test = 100;   % Number of test objects per class
average = 0;

classify = quadrc;  %scenario 1 classifier
%classify = ldc;    %scenario 2 classifier

for i = 1:iter
    % training size scenario 1 = 0.8
    % training size scenario 2 = 0.01
    trainingSize = 0.8;
    %gendat of the data set
    [train, ~] = gendat(nist_data, trainingSize);    
    
    %preprocess the data
    trn_unselected = my_rep1(train);

    %feature selection
    [mapping, R] = pcam(trn_unselected,24);
    
    %mapping the training set
    trn_featsel = trn_unselected*mapping;
    
    %training the classifier
    w = classify(trn_featsel);
    
    %mapping the classifier
    mapped_w = mapping*w;
    
    %evaluation of the classifier with num_test as object count
    e = nist_eval('my_rep1', mapped_w, num_test);
    average = average + e;

end
% print the average result:
average/iter
