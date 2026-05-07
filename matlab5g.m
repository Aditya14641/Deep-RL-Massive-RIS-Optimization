% =========================================================================
% MASSIVE RIS-MISO DEEP REINFORCEMENT LEARNING SIMULATION (MATLAB VERSION)
% Parameters: M=256 (Antennas), K=256 (Users), N=32 (RIS Elements)
% =========================================================================
clear; clc; close all;

%% 1. SYSTEM PARAMETERS
M = 256;          % Number of Base Station Antennas
K = 256;          % Number of Single-Antenna Users
N = 32;           % Number of RIS Elements
Pt_dBm = 20;      % Transmit Power in dBm
Pt_linear = 10^(Pt_dBm/10) / 1000; % Convert to linear Watts

% State and Action Dimensions
% Action: The phase shifts of the N RIS elements (continuous from 0 to 2pi)
numActions = N;
% State: Real and Imaginary parts of the channel matrices (Simplified for speed)
numObservations = 2 * (N * K + M * N + M * K); 

%% 2. CREATE THE ENVIRONMENT
% In MATLAB, environments are created using Action and Observation specifications
obsInfo = rlNumericSpec([numObservations 1]);
obsInfo.Name = 'Channel State Information';

actInfo = rlNumericSpec([numActions 1], 'LowerLimit', 0, 'UpperLimit', 2*pi);
actInfo.Name = 'RIS Phase Shifts';

% We link the custom physics functions (defined at the bottom of this file)
ResetHandle = @() myResetFunction(M, K, N);
StepHandle = @(Action, LoggedSignals) myStepFunction(Action, LoggedSignals, M, K, N, Pt_linear);
env = rlFunctionEnv(obsInfo, actInfo, StepHandle, ResetHandle);

%% 3. BUILD THE NEURAL NETWORKS (512 Neurons)
% --- CRITIC NETWORK ---
% Critic takes both State and Action to calculate the Q-Value (Expected Reward)
statePath = [
    featureInputLayer(numObservations, 'Normalization', 'none', 'Name', 'State')
    fullyConnectedLayer(512, 'Name', 'CriticStateFC1')
    reluLayer('Name', 'CriticRelu1')];

actionPath = [
    featureInputLayer(numActions, 'Normalization', 'none', 'Name', 'Action')
    fullyConnectedLayer(512, 'Name', 'CriticActionFC1')];

commonPath = [
    additionLayer(2, 'Name', 'add')
    reluLayer('Name', 'CriticCommonRelu')
    fullyConnectedLayer(512, 'Name', 'CriticCommonFC1')
    reluLayer('Name', 'CriticCommonRelu2')
    fullyConnectedLayer(1, 'Name', 'CriticOutput')];

criticNetwork = dlnetwork;
criticNetwork = addLayers(criticNetwork, statePath);
criticNetwork = addLayers(criticNetwork, actionPath);
criticNetwork = addLayers(criticNetwork, commonPath);
criticNetwork = connectLayers(criticNetwork, 'CriticRelu1', 'add/in1');
criticNetwork = connectLayers(criticNetwork, 'CriticActionFC1', 'add/in2');

critic = rlQValueFunction(criticNetwork, obsInfo, actInfo, ...
    'ObservationInputNames', 'State', 'ActionInputNames', 'Action');

% --- ACTOR NETWORK ---
% Actor takes State and outputs Action (Phase shifts)
actorNetwork = [
    featureInputLayer(numObservations, 'Normalization', 'none', 'Name', 'State')
    fullyConnectedLayer(512, 'Name', 'ActorFC1')
    reluLayer('Name', 'ActorRelu1')
    fullyConnectedLayer(512, 'Name', 'ActorFC2')
    reluLayer('Name', 'ActorRelu2')
    fullyConnectedLayer(numActions, 'Name', 'ActorActionFC')
    sigmoidLayer('Name', 'ActorSigmoid') % Bounds 0 to 1
    scalingLayer('Name', 'ActorScale', 'Scale', 2*pi) % Scales to 0 to 2pi
];

actorNetwork = dlnetwork(actorNetwork);
actor = rlContinuousDeterministicActor(actorNetwork, obsInfo, actInfo);

%% 4. CONFIGURE THE DDPG AGENT
agentOpts = rlDDPGAgentOptions(...
    'SampleTime', 1, ...
    'TargetSmoothFactor', 1e-3, ...
    'ExperienceBufferLength', 100000, ...
    'MiniBatchSize', 8); % Using the batch size from your Python code

agentOpts.NoiseOptions.Variance = 0.1; 
agentOpts.NoiseOptions.VarianceDecayRate = 1e-5;

agent = rlDDPGAgent(actor, critic, agentOpts);

%% 5. TRAIN THE AI
trainOpts = rlTrainingOptions(...
    'MaxEpisodes', 50, ...                % Force 1 episode like Python script
    'MaxStepsPerEpisode', 1500, ...      % 1500 steps
    'ScoreAveragingWindowLength', 100, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'StopTrainingCriteria', 'EpisodeCount', ...
    'StopTrainingValue', 50);

disp('🚀 Starting MATLAB 256x256 GPU Training...');
% This will open a live graphical plot of the learning curve!
trainingStats = train(agent, env, trainOpts);
disp('🎉 Training Complete!');

% =========================================================================
% ENVIRONMENT PHYSICS FUNCTIONS (Runs every step)
% =========================================================================

function [InitialObservation, LoggedSignals] = myResetFunction(M, K, N)
    % Generates the initial random Rayleigh Fading Channels
    % H_d: Direct channel (BS to Users) M x K
    % H_r: Reflected channel (RIS to Users) N x K
    % G: BS to RIS channel M x N
    
    LoggedSignals.H_d = (randn(M, K) + 1i*randn(M, K)) / sqrt(2);
    LoggedSignals.H_r = (randn(N, K) + 1i*randn(N, K)) / sqrt(2);
    LoggedSignals.G = (randn(M, N) + 1i*randn(M, N)) / sqrt(2);
    
    % Flatten into a 1D Real array for the Neural Network
    state_complex = [LoggedSignals.H_d(:); LoggedSignals.H_r(:); LoggedSignals.G(:)];
    InitialObservation = [real(state_complex); imag(state_complex)];
end

function [Observation, Reward, IsDone, LoggedSignals] = myStepFunction(Action, LoggedSignals, M, K, N, Pt)
    % 1. Rebuild the Observation from the LoggedSignals struct!
    % This flattens the matrices back into the 163,840x1 numeric array 
    % that the Neural Network expects, fixing the validation error.
    state_complex = [LoggedSignals.H_d(:); LoggedSignals.H_r(:); LoggedSignals.G(:)];
    Observation = double([real(state_complex); imag(state_complex)]);
    
    % 2. Calculate the AI's Reward
    % (Mock calculation for architectural testing)
    optimal_alignment = pi * ones(N, 1); 
    phase_error = sum(abs(Action - optimal_alignment));
    Reward = double(10 * log10(Pt * 1000) - phase_error); 
    
    % 3. The environment never ends early in this setup
    IsDone = false; 
end