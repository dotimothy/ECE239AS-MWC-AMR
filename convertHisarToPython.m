% Conversion Script from Hisar Data CSV to .mat file to read in Python 
clc; close all; clear;

% Change to Wherever the Original Data CSVs Are %
trainDir = './datasets/HisarMod2019.1/HisarMod2019.1/Train';
trainCSVDir = strcat(trainDir,'/train_data.csv');
trainMatDir = strcat(trainDir,'/train_data.mat');
testDir = './datasets/HisarMod2019.1/HisarMod2019.1/Test';
testCSVDir = strcat(testDir,'/test_data.csv');
testMatDir = strcat(testDir,'/test_data.mat');


% Converting Train Data to IQ Matrix then save to .mat file
if not(isfile(trainMatDir))
    train_data = readmatrix(trainCSVDir);
    train_data = cat(3,real(train_data),imag(train_data));
    save(trainMatDir,'train_data','-v7.3');
    clear train_data;
end

% Converting Test Data to IQ Matrix then save to .mat file
if not(isfile(testMatDir))
    test_data = readmatrix(testCSVDir);
    test_data = cat(3,real(test_data),imag(test_data));
    save(testMatDir,'test_data','-v7.3');
    clear test_data;
end
