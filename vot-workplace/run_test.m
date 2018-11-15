% This script can be used to test the integration of a tracker to the
% framework.

addpath('C:\Users\Barovir\PycharmProjects\diplomska-naloga\vot-toolkit'); toolkit_path; % Make sure that VOT toolkit is in the path

[sequences, experiments] = workspace_load();

tracker = tracker_load('LT');

workspace_test(tracker, sequences);

