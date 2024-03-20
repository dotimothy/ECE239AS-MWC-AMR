% clc; close all; clear;
% 
% dataDir = './datasets/HisarMod2019.1/HisarMod2019.1/';
% testDataDir = strcat(dataDir,'Test/test_data.csv');
% testData = readmatrix(testDataDir);

index = 10;
figure;
hold on; 
scatter(real(testData(index,:)),imag(testData(index,:)));
grid on;