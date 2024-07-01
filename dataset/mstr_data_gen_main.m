clear all; close all;clc;
input_file_name = 'recons_shapes.mat';
output_file_name = 'recons_matlab_mstr_homog.mat';
generate_homogenized_data(input_file_name, output_file_name);
