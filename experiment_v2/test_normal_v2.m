datadir = "/home/home02/mm23rn/resnet_18_v1"; 
downloadCIFARData(datadir);

[~,~,XValidation,TValidation] = loadCIFARData(datadir);

classes = categories(TValidation);

imageSize = [224 224 3];

augimdsVal = augmentedImageDatastore(imageSize,XValidation,TValidation);

load("resnet_18_v2.mat", "netFGSM");

miniBatchSize = 512;

mbqVal = minibatchqueue(augimdsVal, ...
    MiniBatchSize=miniBatchSize,...
    MiniBatchFcn=@preprocessMiniBatch,...
    MiniBatchFormat=["SSCB",""]);

YPred = modelPredictions(netFGSM,mbqVal,classes);
acc = mean(YPred == TValidation);

fileID = fopen('nrml_acc_resnet_18_v2.txt', 'w');

val_acc = "Validation accuracy: " + acc*100 + "%";

fprintf(fileID, val_acc);

quit;

%--------------------------SUPPORTING FUNCTIONS---------------------------------------

function [X,T] = preprocessMiniBatch(XCell,TCell)

% Concatenate.
X = cat(4,XCell{1:end});

X = single(X);

% Extract label data from the cell and concatenate.
T = cat(2,TCell{1:end});

% One-hot encode labels.
T = onehotencode(T,1);

end

function predictions = modelPredictions(net,mbq,classes)

predictions = [];

while hasdata(mbq)

    XTest = next(mbq);
    YPred = predict(net,XTest);

    YPred = onehotdecode(YPred,classes,1)';

    predictions = [predictions; YPred];
end

end

function downloadCIFARData(destination)

url = 'https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz';

unpackedData = fullfile(destination,'cifar-10-batches-mat');
if ~exist(unpackedData,'dir')
    fprintf('Downloading CIFAR-10 dataset (175 MB). This can take a while...');
    untar(url,destination);
    fprintf('done.\n\n');
end

end

function [XTrain,YTrain,XTest,YTest] = loadCIFARData(location)

location = fullfile(location,'cifar-10-batches-mat');

[XTrain1,YTrain1] = loadBatchAsFourDimensionalArray(location,'data_batch_1.mat');
[XTrain2,YTrain2] = loadBatchAsFourDimensionalArray(location,'data_batch_2.mat');
[XTrain3,YTrain3] = loadBatchAsFourDimensionalArray(location,'data_batch_3.mat');
[XTrain4,YTrain4] = loadBatchAsFourDimensionalArray(location,'data_batch_4.mat');
[XTrain5,YTrain5] = loadBatchAsFourDimensionalArray(location,'data_batch_5.mat');
XTrain = cat(4,XTrain1,XTrain2,XTrain3,XTrain4,XTrain5);
YTrain = [YTrain1;YTrain2;YTrain3;YTrain4;YTrain5];

[XTest,YTest] = loadBatchAsFourDimensionalArray(location,'test_batch.mat');
end

function [XBatch,YBatch] = loadBatchAsFourDimensionalArray(location,batchFileName)
s = load(fullfile(location,batchFileName));
XBatch = s.data';
XBatch = reshape(XBatch,32,32,3,[]);
XBatch = permute(XBatch,[2 1 3 4]);
YBatch = convertLabelsToCategorical(location,s.labels);
end

function categoricalLabels = convertLabelsToCategorical(location,integerLabels)
s = load(fullfile(location,'batches.meta.mat'));
categoricalLabels = categorical(integerLabels,0:9,s.label_names);
end
