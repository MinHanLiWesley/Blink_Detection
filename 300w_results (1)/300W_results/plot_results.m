function plot_results(version, x_limit, colors, linewidth, fontsize)
% PLOT_RESULTS  Generates the 300W Faces In-The-Wild Challenge (300-W)
% results in the form of Cumulative Error Distributions (CED) curves. The
% function renders the indoor, outdoor and indoor + outdoor results based
% on both 68 and 51 landmark points in 6 different figures.
%
% Please cite:
% C. Sagonas, E. Antonakos, G. Tzimiropoulos, S. Zafeiriou, M. Pantic. "300
% Faces In-The-Wild Challenge: Database and Results", Image and Vision
% Computing, 2015.
%
%   PLOT_RESULTS(VERSION) plots the results of the specified 
%   challenge's version. If VERSION is 1, then the results of the 300W 2013 
%   ICCV Workshop are returned. If VERSION is 2, then the results of the 
%   300W 2015 IMAVIS Special Issue are returned.
%
%   PLOT_RESULTS(VERSION,X_LIMIT) where X_LIMIT is a float that 
%   specifies the maximum value of the horizontal axis with the errors. 
%   Default is 0.08.
%
%   PLOT_RESULTS(VERSION,X_LIMIT,COLORS) where COLORS is an Nx3 matrix
%   with an RGB color value for each of the N curves. The colors are 
%   sampled from the lines colormap by default.
%
%   PLOT_RESULTS(VERSION,X_LIMIT,COLORS,LINEWIDTH) where LINEWIDTH is 
%   a float that specifies the line width of the rendered CED curves. 
%   Default is 3.
%
%   PLOT_RESULTS(VERSION,X_LIMIT,COLORS,LINEWIDTH,FONTSIZE) where 
%   FONTSIZE defines the font size of the axes and legend. Default is 12.


% Cell of filenames with the CED values of each participant
if version == 1
    participants = {'Baltrusaitis', 'Hasan', 'Jaiswal', 'Milborrow', 'Yan', 'Zhou'};
    fprintf('Plotting results of 300W Challenge 2013... ');
elseif version == 2
    participants = {'Cech', 'Deng', 'Fan', 'Martinez', 'Uricar'};
    fprintf('Plotting results of 300W Challenge 2015... ');
else
    error('version must be either 1 or 2');
end
numberOfParticipants = length(participants);

% Check arguments
if nargin < 2
    x_limit = 0.08;
end
if nargin < 3
    colors = lines(numberOfParticipants);
end
if nargin < 4
    linewidth = 3;
end
if nargin < 5
    fontsize = 12;
end

% Initialize matrices
ced68 = zeros(701, numberOfParticipants);
ced68_indoor = zeros(701, numberOfParticipants);
ced68_outdoor = zeros(701, numberOfParticipants);
ced51 = zeros(701, numberOfParticipants);
ced51_indoor = zeros(701, numberOfParticipants);
ced51_outdoor = zeros(701, numberOfParticipants);
legend_entries = cell(1, numberOfParticipants);

% Load results
results_path = ['300W_v', num2str(version)];
for i=1:numberOfParticipants
    % Open and read file
    fid = fopen(fullfile(results_path, [participants{i}, '.txt']), 'r');
    tmp = textscan(fid, '%f %f %f %f %f %f %f', 'Headerlines', 4);
    fclose(fid);
    % Get CED values
    bins = tmp{1};
    ced68(:, i) = tmp{2};
    ced68_indoor(:, i) = tmp{3};
    ced68_outdoor(:, i) = tmp{4};
    ced51(:, i) = tmp{5};
    ced51_indoor(:, i) = tmp{6};
    ced51_outdoor(:, i) = tmp{7};
    % Update legend entries
    legend_entries{i} = [participants{i}, ' et al.'];
end

% Render curves
plot_ced(bins, ced68, legend_entries, 'Indoor + Outdoor, 68 points', ...
         x_limit, colors, linewidth, fontsize);
plot_ced(bins, ced68_indoor, legend_entries, 'Indoor, 68 points', ...
         x_limit, colors, linewidth, fontsize);
plot_ced(bins, ced68_outdoor, legend_entries, 'Outdoor, 68 points', ...
         x_limit, colors, linewidth, fontsize);
plot_ced(bins, ced51, legend_entries, 'Indoor + Outdoor, 51 points', ...
         x_limit, colors, linewidth, fontsize);
plot_ced(bins, ced51_indoor, legend_entries, 'Indoor, 51 points', ...
         x_limit, colors, linewidth, fontsize);
plot_ced(bins, ced51_outdoor, legend_entries, 'Outdoor, 51 points', ...
         x_limit, colors, linewidth, fontsize);
     
fprintf('done.\n');

end


function plot_ced(bins, ced_values, legend_entries, title_str, x_limit, ...
                  colors, linewidth, fontsize)
% Check arguments
numberOfCurves = size(ced_values, 2);
if nargin < 5
    x_limit = 0.08;
end
if nargin < 6
    colors = colormap(lines(numberOfCurves));
end
if nargin < 7
    linewidth = 2;
end
if nargin < 8
    fontsize = 12;
end

% Render curves
figure;
for k=1:numberOfCurves
    plot(bins, ced_values(:, k), ...
         'color', colors(k, :), ...
         'linewidth', linewidth, ...
         'marker', 'none', ...
         'linestyle', '-');
    hold on;
end
hold off;

% Enable grid
grid on;
set(gca, 'gridlinestyle', '--');
set(gca, 'fontsize', fontsize);
set(gca, 'ytick', 0:0.1:1);

% Set labels, limits and legend
ylabel('Images Proportion', 'fontsize', fontsize);
xlabel('Point-to-point Normalized RMS Error', 'fontsize', fontsize); 
title(title_str, 'fontsize', fontsize);
xlim([0, x_limit]);
ylim([0.0, 1.0]);
legend(legend_entries, 'Location', 'NorthWest');

end
