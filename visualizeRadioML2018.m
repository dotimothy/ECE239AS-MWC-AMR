clc; close all; clear;

dataDir = 'RadioML/2018.01.OSC.0001_1024x2M.h5/2018.01/GOLD_XYZ_OSC.0001_1024.hdf5';
h5disp(dataDir,'/Y');

index = 933989;
data = h5read(dataDir,'/X',[1 1 index],[2 1024 1],[1 1 1]);
classes = {'32PSK','16APSK','32QAM','FM','GMSK','32APSK','OQPSK','8ASK','BPSK','8PSK','AM-SSB-SC','4ASK','16PSK','64APSK','128QAM','128APSK','AM-DSB-SC','AM-SSB-WC','64QAM','QPSK','256QAM','AM-DSB-WC','OOK','16QAM'};
y = h5read(dataDir,'/Y',[1 index],[24 1],[1 1]);
Y = h5read(dataDir,'/Y');
Z = h5read(dataDir,'/Z');
Y = onehotdecode(Y,classes,1);
idx = find(Y =='BPSK' & Z == 20);
class =  Y(index);
z = h5read(dataDir,'/Z',[1 index],[1 1],[1 1]);
x = data(1,:) + 1i*data(2,:);
figure; 
hold on; 
grid on;
scatter(real(x),imag(x));
ax = gca;
ax.XAxisLocation = 'origin';
ax.YAxisLocation = 'origin';