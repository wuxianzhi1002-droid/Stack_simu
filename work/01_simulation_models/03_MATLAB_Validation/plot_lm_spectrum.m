%% Scientific Plotting of Lumerical Spectrum
% This script reads lm_spectrum.txt and plots the reflection spectrum
% with a standard research-style format.

clear; clc; close all;

%% 1. Data Loading
filename = '2um_cavity_plus_998um.txt';
% filename = '17-6um_cavity.txt';
% filename = '0-1um_cavity.txt';
% Read data skipping the first header line
data = readmatrix(filename, 'NumHeaderLines', 1);

%% 2. Coordinate Setup
% Wavelength range: 0.4 to 0.8 um
num_points = length(data);
lambda_um = linspace(0.4, 0.8, num_points)';

%% 3. Plotting (Scientific Style)
figure('Color', 'w', 'Units', 'inches', 'Position', [2, 2, 6, 4]);

plot(lambda_um, data, 'Color', [0, 0.447, 0.741], 'LineWidth', 1.5);

% Formatting
ax = gca;
ax.FontSize = 11;
ax.FontName = 'Arial';
ax.LineWidth = 1.2;
ax.Box = 'on';
ax.TickDir = 'in';
ax.XMinorTick = 'on';
ax.YMinorTick = 'on';

% Labels
xlabel('Wavelength (\mum)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Reflected Power (a.u.)', 'FontSize', 12, 'FontWeight', 'bold');
title('Measured Reflection Spectrum (FDTD Monitor)', 'FontSize', 13);

% Axis Limits
xlim([0.4, 0.8]);
grid on;
ax.GridLineStyle = ':';
ax.GridAlpha = 0.5;

% Save result
saveas(gcf, 'lm_spectrum_plot.png');
fprintf('Plotting Complete. Data points: %d\n', num_points);
