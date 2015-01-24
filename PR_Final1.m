%Matlab final Assignment Pattern Recognition:

%Load Library (dipImage)
clc; clear all; close all;

%Load data
data = prnist([0:9], [1:1000]);
prwaitbar off;
prwarning off
%convert data set + preprocessing
imageSize = [64 64];
preProcessing= im_box([],0,1)*im_gauss;
preProcessing = preProcessing* im_resize([],imageSize);
preProcessing = preProcessing*im_box([],1,0);
dataSetResized = data*preProcessing;
finalImgDataSet = prdataset(dataSetResized)

%feature extraction
dataSetWithFeatures = im_features(finalImgDataSet, 'all');
featureSize = length(dataSetWithFeatures.featlab)

%Classifiers:
%Nearest Mean Classifier, Linear Bayes Normal Classifier,
%Quadratic Bayes Normal Classifier, Fisher's Least Square Linear Classifier
%Logistic Linear Classifier
parametric = {nmc,ldc,qdc,fisherc,loglc};
%K-Nearest Neighbor Classifier, Parzen classifier
non_parametric = {knnc,parzenc};
%Decision Tree Classifier
%Back-propagation trained feed-forward neural net classifier
advanced = {dtc, bpxnc};

%split data set for training and testing
trainingSize = 0.01;
[train, test] = gendat(dataSetWithFeatures, trainingSize);

%select feature data
featCrit = { 'NN', 'maha-m', 'eucl-m', };
critName = char(featCrit(1));
[wNNI, rNNI] = featseli(train, critName, featureSize);
trainedFeaturesNNI = train * wNNI;
testFeaturesNNI = test * wNNI;
[wNNF, rNNF] = featself(train, critName, featureSize);
trainedFeaturesNNF = train * wNNF;
testFeaturesNNF = test * wNNF;
[wNNB, rNNB] = featselb(train, critName, featureSize);
trainedFeaturesNNB = train * wNNB;
testFeaturesNNB = test * wNNB;
[wNNP, rNNP] = featselp(train, critName, featureSize);
trainedFeaturesNNP = train * wNNP;
testFeaturesNNP = test * wNNP;


critName = char(featCrit(2));
[wMahaI, rMahaI] = featseli(train, critName, featureSize);
trainedFeaturesMahaI = train * wMahaI;
testFeaturesMahaI = test * wMahaI;
[wMahaF, rMahaF] = featself(train, critName, featureSize);
trainedFeaturesMahaF = train * wMahaF;
testFeaturesMahaF = test * wMahaF;
[wMahaB, rMahaB] = featselb(train, critName, featureSize);
trainedFeaturesMahaB = train * wMahaB;
testFeaturesMahaB = test * wMahaB;
[wMahaP, rMahaP] = featselp(train, critName, featureSize);
trainedFeaturesMahaP = train * wMahaP;
testFeaturesMahaP = test * wMahaP;

critName = char(featCrit(3));
[wEuclI, rEuclI] = featseli(train, critName, featureSize);
trainedFeaturesEuclI = train * wEuclI;
testFeaturesEuclI = test * wEuclI;
[wEuclF, rEuclF] = featself(train, critName, featureSize);
trainedFeaturesEuclF = train * wEuclF;
testFeaturesEuclF = test * wEuclF;
[wEuclB, rEuclB] = featselb(train, critName, featureSize);
trainedFeaturesEuclB = train * wEuclB;
testFeaturesEuclB = test * wEuclB;
[wEuclP, rEuclP] = featselp(train, critName, featureSize);
trainedFeaturesEuclP = train * wEuclP;
testFeaturesEuclP = test * wEuclP;

for i = length(parametric):-1:1
    
end
featnum = 1:featureSize;
e1 = clevalf(trainedFeaturesEuclI,ldc,featnum,[],1,trainedFeaturesEuclI);
e2 = clevalf(trainedFeaturesEuclF,ldc,featnum,[],1,trainedFeaturesEuclF);
e3 = clevalf(trainedFeaturesEuclB,ldc,featnum,[],1,trainedFeaturesEuclB);
e4 = clevalf(trainedFeaturesEuclP,ldc,featnum,[],1,trainedFeaturesEuclP);
figure
plote({e1,e2,e3,e4})
legend({'I','F','B','P'})

%Train classifier

%Test classifier

%Report performance for both