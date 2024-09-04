function [EI, wAmp, wDNF] = calculateFEI(Signal, windowSize, windowOverlap)
% Originally created by Richard Hardstone (2020), rhardstone@gmail.com
% Please note that commercial use of this algorithm is protected by Patent claim (PCT/NL2019/050167) .Method of determining brain activity.; with priority date 16 March 2018
% This code is licensed under creative commons license CC-BY-NC https://creativecommons.org/licenses/by-nc/4.0/legalcode.txt

%%
% Calculates fEI (on a set window size) for signal
% Steps refer to description of fEI algorithm in Figure 2D of paper:
%   Measurement of excitation inhibition ratio in autism spectrum disorder using critical brain dynamics
%   Scientific Reports (2020)
%   Hilgo Bruining*, Richard Hardstone*, Erika L. Juarez-Martinez*, Jan Sprengers*, Arthur-Ervin Avramiea, Sonja Simpraga, Simon J. Houtman, Simon-Shlomo Poil5,
%   Eva Dallares, Satu Palva, Bob Oranje, J. Matias Palva, Huibert D. Mansvelder & Klaus Linkenkaer-Hansen
%   (*Joint First Author)
%
%

%%Input
% Signal:        amplitude envelope with dimensions (numSamples,numChannels)
% windowSize:    in samples
% windowOverlap: is fraction of overlap between windows (0-1)

%% Example:
% [EI, wAmp, wDNF] = calculateFEI(randn(100000,1), 5000, 0.8)
% Will calculate fEI on the single channel white noise signal with length 100000
% using a window size of 5000 samples, and 80% overlap between each window

lengthSignal = size(Signal,1);
numChannels = size(Signal,2);

windowOffset = floor(windowSize * (1-windowOverlap));
allWindowIndex = createWindowIndices(lengthSignal, windowSize, windowOffset);
numWindows = size(allWindowIndex,1);

EI = zeros(numChannels,1);
wAmp = zeros(numChannels,numWindows);
wDNF = zeros(numChannels,numWindows);

for i_channel = 1:numChannels
    originalAmplitude = Signal(:,i_channel);    
    signalProfile = cumsum(originalAmplitude - mean(originalAmplitude));            % Step ii -> iii
    w_originalAmplitude = mean(originalAmplitude(allWindowIndex),2);                % Calculate mean amplitude for each window
    xAmp = repmat(w_originalAmplitude,[1 windowSize]);
    xSignal = signalProfile(allWindowIndex);                                        % Arrange Signals into windows
    xSignal = (xSignal ./ xAmp)';                                                   % Step iii -> iv
    dSignal = detrend(xSignal);                                                     % Step iv -> v
    w_detrendedNormalizedFluctuations = std(dSignal,1)';                            % Step v -> vi
    EI(i_channel)  = 1-corr(w_detrendedNormalizedFluctuations,w_originalAmplitude); % Step vi -> vii
    wAmp(i_channel,:) = w_originalAmplitude;
    wDNF(i_channel,:) = w_detrendedNormalizedFluctuations;
end


function allWindowIndex = createWindowIndices(lengthSignal, lengthWindow, windowOffset)
%Gets indices for a set of windows of size (lengthWindow) with a set overlap between them

windowStarts = (1:windowOffset:lengthSignal-lengthWindow+1)-1;
numWindows = length(windowStarts);

oneWindowIndex = 1:lengthWindow;
allWindowIndex = repmat(oneWindowIndex,[numWindows,1]);

allWindowIndex = allWindowIndex + repmat(windowStarts',[1 lengthWindow]);
