% This script can be used to interactively inspect the results

addpath('C:\Users\Barovir\PycharmProjects\diplomska-naloga\vot-toolkit'); toolkit_path; % Make sure that VOT toolkit is in the path

[sequences, experiments] = workspace_load();

trackers = tracker_load('NCC');

workspace_browse(trackers, sequences, experiments);

