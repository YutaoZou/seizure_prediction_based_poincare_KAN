% ---Slicing and feature extraction were performed on the CHB-MIT and Huashan datasets 
% that have undergone filtering and separation of positive and negative samples---
clc,clear;

% Sampling rate
sampling_rate = 256;
% Overlap rate
overlap = 0.7;
% Sliding window length
window_length = 4;

% Obtain the file addresses of positive and negative sample data
p_file_path = 'F:\CHB_positive_negative_data\positive';
n_file_path = 'F:\CHB_positive_negative_data\negative';

% Save the position of the 5-fold cross validation feature set
save_cross_validation_set_path = 'E:\BaiduSyncdisk\EEG\Chb_data';

% Obtain file data based on the address
p_mat_list = dir(fullfile(p_file_path,'*.mat'));
n_mat_list = dir(fullfile(n_file_path,'*.mat'));


% Screen out the data of the same cases and put them together

% The number of files for obtaining the original data of positive and negative samples
p_mat_nums = size(p_mat_list, 1);
n_mat_nums = size(n_mat_list, 1);


patients_num = 24;
pats_seizure_nums = cell(patients_num,3);

i = 1;
samples_nums_per_pat = [];

% Positive sample data slicing

for p_mat_index = 1:p_mat_nums-1

    % Take the first 5 characters
    patient_1_name = p_mat_list(p_mat_index).name(1:5);
    patient_2_name = p_mat_list(p_mat_index+1).name(1:5);

    % Determine whether the previous and subsequent documents belong to the same case
    if strcmp(patient_1_name, patient_2_name)

        % Loading data begins
        loaddata_p_1 = load(fullfile(p_file_path, p_mat_list(p_mat_index).name));
        loaddata_p_1 = loaddata_p_1.data;

        % Slicing begins
        output_cell_1 = slice_sampling(loaddata_p_1, window_length, sampling_rate, overlap);
        
        pats_seizure_nums{i,1}{end+1,1} = output_cell_1;
        samples_num = length(output_cell_1);
        samples_nums_per_pat = [samples_nums_per_pat,samples_num];
    else
        % Loading data begins
        loaddata_p_2 = load(fullfile(p_file_path, p_mat_list(p_mat_index).name));
        loaddata_p_2 = loaddata_p_2.data;

        % Slicing begins
        output_cell_2 = slice_sampling(loaddata_p_2, window_length, sampling_rate, overlap);
        
        pats_seizure_nums{i,1}{end+1,1} = output_cell_2;
        samples_num = length(output_cell_2);
        samples_nums_per_pat = [samples_nums_per_pat,samples_num];

        pats_seizure_nums{i,2} = samples_nums_per_pat;

        i = i + 1;          
        samples_nums_per_pat = [];
    end

    if p_mat_index == p_mat_nums-1
        % Loading data begins
        loaddata_p_2 = load(fullfile(p_file_path, p_mat_list(p_mat_index+1).name));
        loaddata_p_2 = loaddata_p_2.data;

        % Slicing begins
        output_cell_2 = slice_sampling(loaddata_p_2, window_length, sampling_rate, overlap);
        
        pats_seizure_nums{i,1}{end+1,1} = output_cell_2;
        samples_num = length(output_cell_2);
        samples_nums_per_pat = [samples_nums_per_pat,samples_num];

        pats_seizure_nums{i,2} = samples_nums_per_pat;
    end

end

%% The slicing of negative sample data begins

% Pre-treatment: Negative documents are grouped by patients
n_patient_files = cell(patients_num, 1);
for i = 1:patients_num
    patient_id = sprintf('chb%02d', i);
    % Precisely match the patient ID (processing file names such as chb01_06_negative_1.mat)
    mask = cellfun(@(x) startsWith(x, patient_id), {n_mat_list.name});
    n_patient_files{i} = n_mat_list(mask);
end

% The main circulation processes each patient
for i = 1:patients_num
    % The demand for obtaining positive samples
    pos_counts = pats_seizure_nums{i, 2};
    total_neg = sum(pos_counts);
    
    % Initialize the storage of negative samples
    neg_samples = cell(total_neg, 1);
    current_idx = 1;
    
    % Obtain all negative documents of the current patient
    files = n_patient_files{i};
    if isempty(files)
        error('Patient chb%02d has no negative documents', i);
    end
    file_list = {files.name};
    
    % Sampling tasks are assigned according to the number of seizure
    for epoch = 1:length(pos_counts)
        required = pos_counts(epoch);
        remaining = required;
        
        while remaining > 0
            
            [~, file_idx] = histc(rand, linspace(0, 1, length(file_list)+1));
            selected_file = fullfile(n_file_path, file_list{file_idx});
            
            
            data = load(selected_file);
            signal = data.data; 
            [~, n_points] = size(signal);
            
            % Calculate the slice parameters
            slice_length = 4 * sampling_rate;
            max_start = n_points - slice_length + 1;
            
            if max_start <= 0
                continue; 
            end
            
            % Generate random starting points (overlapping is allowed)
            starts = randi([1, max_start], [1, remaining]);
            unique_starts = unique(starts); % Deduplication avoids repeated sampling of the same position
            
            % Extract valid samples
            for s = 1:length(unique_starts)
                start_idx = unique_starts(s);
                end_idx = start_idx + slice_length - 1;
                neg_samples{current_idx} = signal(:, start_idx:end_idx);
                current_idx = current_idx + 1;
            end
            
            % Update the remaining requirements
            remaining = remaining - length(unique_starts);
        end
        clear signal;
    end
    
    % Store the result
    pats_seizure_nums{i, 3} = neg_samples;
end


%% Extract features

for m = 1:24

    seizure_nums = length(pats_seizure_nums{m,1});

    % Negative sample feature extraction 
    negative_X = extract_geometric_features(pats_seizure_nums{m,3});

    % Feature extraction of positive samples
    positive_X = [];

    for n = 1:seizure_nums
        p_features_per_seizure = extract_geometric_features(pats_seizure_nums{m,1}{n});

        positive_X = [positive_X; p_features_per_seizure];
    end

    % Positive and negative samples are labeled and randomly integrated
    data_combined_shuffled = combineAndShuffleData(negative_X, positive_X);
    save(fullfile(save_cross_validation_set_path, sprintf('data_combined_shuffled_%d.mat',m)),...
        'data_combined_shuffled');

end

%%

function output_cell = slice_sampling(positive_data, window_length, sampling_rate, overlap)

    output_cell = [];
    % Calculate the time step (in seconds)
    step_time = window_length * (1 - overlap); % Calculate the step size based on the overlap ratio
    % Calculate the slice parameters
    total_samples = size(positive_data, 2);    % Total sample size
    win_samples = round(window_length * sampling_rate); % Number of window samples (rounded off)
    step_samples = round(step_time * sampling_rate);   % Step size sample size (rounded off)
    
    % Calculate the maximum number of slicable slices
    max_start = total_samples - win_samples; % Maximum starting position
    num_slices = floor(max_start / step_samples) + 1; % Slicing quantity
    
    % Generate the slice starting index (starting from 1)
    slice_indices = (0:num_slices-1)*step_samples + 1;
    
    % Perform the slicing operation
    slices = arrayfun(@(x) positive_data(1:18, x:x+win_samples-1),...
                      slice_indices, 'UniformOutput', false);
    output_cell = [output_cell; slices'];
end

function data_combined_shuffled = combineAndShuffleData(negative_X, positive_X)
    % Add a label
    labels_negative = zeros(size(negative_X, 1), 1);
    labels_positive = ones(size(positive_X, 1), 1);

    % Label and feature integration
    data_negative = [negative_X, labels_negative];
    data_positive = [positive_X, labels_positive];

    % Integration of positive and negative samples
    data_combined = [data_negative; data_positive];
    idx_random = randperm(size(data_combined, 1));
    data_combined_shuffled = data_combined(idx_random, :);
end
