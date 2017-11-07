clc
close all
clear all

fileID = fopen('azimuth_image.txt','r');
A = fscanf(fileID,'%f');
A1=reshape(A,[1024,2048]);
imagesc((A1))
