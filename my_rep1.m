function a = my_rep1( m )
    % Preprocess the digits
    imageSize = [32 32];
    preProcessing= im_box([],0,1)*im_gauss(2,2);
    preProcessing = preProcessing* im_resize([],imageSize);
    preProcessing = preProcessing*im_box([],1,0);
    dataSetResized = m*preProcessing;
    a = prdataset(dataSetResized);
end