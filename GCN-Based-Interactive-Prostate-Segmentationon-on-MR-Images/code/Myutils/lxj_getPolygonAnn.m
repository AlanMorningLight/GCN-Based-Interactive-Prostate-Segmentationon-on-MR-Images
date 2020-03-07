clear all
close all
clc

I = imread('RETOUCH_case0004_0096.png');

threshold = graythresh(I);
BW=im2bw(I,threshold);
% BW = imbinarize(I);
[B,L] = bwboundaries(BW,'noholes');
imshow(label2rgb(L, @jet, [.5 .5 .5]))
hold on
% for k = 1:length(B)
for k = 1:1
    boundary = B{k};
    for mm = 1:length(boundary)
        vi = length(boundary)
        b1 = boundary(mm,2)
        b2 = boundary(mm,1)
        plot(boundary(mm,2), boundary(mm,1), 'w', 'LineWidth', 5)
    end
    
end