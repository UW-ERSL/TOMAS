function  generate_homogenized_data(data_file, output_file)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Args:
% data_file      = File which has an array of size (no. of examples, nelx, nely).
%                  For each example we have (nelx, nely) pixel values.
%                  The pixel values are either 0 or 1, where 0 corresponds to
%                  solid pixel and 1 corresponds to void/fluid pixel.
% output_file    =  Name of output file which stores homogenized data of
%                  size (no. of examples, 4). 4 corresponds to no. of
%                  components of homogenized tensor.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Load data
data_struct = load(data_file);
% data = permute(data_struct.mstr_images, [1, 3, 2]);
data = data_struct.mstr_images;

%% Parameters
lx = 1; % length of microstr along x
ly = 1; % length of microstr along y
solidPermeability = 1e6;
fluidPermeability = 0;
matProp = [solidPermeability fluidPermeability];
cellAngleDeg = 90; % Oreintation of the mstr wrt cell axis
[examples, nelx, nely] = size(data);

%% initalization
mstr = zeros(examples,1);
c00 = zeros(examples,1);
c11 = zeros(examples,1);
c01 = zeros(examples,1);
c10 = zeros(examples,1);


for i = 1:examples
    allOnes = all(data(i, :, :) == 1);

    % Display the result
    if allOnes
        data(i, (nelx)/2, (nelx)/2) = 0;
    end
    current_mstr = transpose(reshape(data(i, :, :), nelx , nely));
    ch = fluidHomogenization(lx,ly,matProp,cellAngleDeg, current_mstr);
    mstr(i) = i;  
    c00(i) = ch(1,1);
    c11(i) = ch(2,2);
    c01(i) = ch(1,2);
    c10(i) = ch(2,1);
    disp('--- mstr number ---'); disp(i)
end

% Create a struct to store the variables
homogen_data = struct('mstr', mstr, 'c00', c00, 'c11', c11, 'c01', c01, 'c10', c10);

% Save the variables to a .mat file
save(output_file, '-struct', 'homogen_data');
