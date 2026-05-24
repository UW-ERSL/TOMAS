% Run fluid homogenization on the 7000-sample dataset (dataset_num=2).
% Writes dataset/homogen_data_2.mat with fields mstr, c00, c11, c01, c10.
clear all; close all; clc;

% Make sure helper .m files are on the path.
this_dir = fileparts(mfilename('fullpath'));
addpath(this_dir);

input_file = fullfile(this_dir, 'mstr_images_2.mat');
output_file = fullfile(this_dir, 'homogen_data_2.mat');

fprintf('Input  : %s\n', input_file);
fprintf('Output : %s\n', output_file);

start_time = tic;
generate_homogenized_data(input_file, output_file);
elapsed = toc(start_time);
fprintf('\nHomogenization finished in %.1f minutes.\n', elapsed / 60);
