close all

mrstModule add ad-core mrst-gui 

paramobj       = ReactionDiffusionInputParams([]);
paramobj.k_1   = 4e6*(1/(mol/litre))*(1/second); % reaction on
paramobj.k_2   = 5*(1/second); % reaction off
paramobj.N.D   = 8e-7*(meter^2/second); % diffusion constant
paramobj.R.D   = 0*(meter^2/second); % the reactors don't diffuse
paramobj.R_N.D = 0*(meter^2/second);

% creating the synapse model grid
G = Cylindergrid();
G = computeGeometry(G);

paramobj.G = G;

paramobj = paramobj.validateInputParams();

model = ReactionDiffusion(paramobj);

% setup the simulation timings
total = 5e-4*second;
n     = 100;
dt    = total/n;
step  = struct('val', dt*ones(n, 1), 'control', ones(n, 1));

control.none = [];
schedule = struct('control', control, 'step', step);

% setup the systems initial state
G = model.G;
receptorCells = (10963 : 12180); % reaction layer
indexs = find(G.cells.volumes(1:1218) < 1e-27); % getting the central cells of the top layer
injectionCells = indexs; % first layer

% if we want to see the systems initial state
doplot = false;
if doplot
    figure
    plotGrid(G, 'facecolor', 'none');
    plotGrid(G, injectionCells, 'facecolor', 'yellow');
    plotGrid(G, receptorCells, 'facecolor', 'blue');
    title("Initial System")
    view(33, 26);
    return
end

% more initial state setup

A = 6.02214076e23; % Avogadro constant

nc     = G.cells.num; % total number of cells
vols   = G.cells.volumes; % total volume of synapse

initCR = (1000/A)*((micro*meter)^2)/sum(G.cells.volumes(receptorCells)); % concentration of initial receptor cells

V      = sum(G.cells.volumes(injectionCells)); % volume of cells starting with neurotransmitters
initCN = (2000/A)/V;

cR                 = zeros(nc, 1);
cR(receptorCells)  = initCR;
cN                 = zeros(nc, 1);
cN(injectionCells) = initCN;
cR_N               = zeros(nc, 1);

initstate.R.c   = cR;
initstate.N.c   = cN;
initstate.R_N.c = cR_N;

% run simulation

nls = NonLinearSolver();
nls.errorOnFailure = false;

[~, states, report] = simulateScheduleAD(initstate, model, schedule, 'NonLinearSolver', nls);

%%

% take out states that might have not converged
ind = cellfun(@(state) ~isempty(state), states);
states = states(ind);

% the figures we will plot
figure(1); figure(2); figure(3); figure(4); figure(5)

C_R_vec = zeros(n,1);
C_RN_vec = zeros(n,1);

% investigate the system after 3 time steps
state = states{3};

set(0, 'currentfigure', 5);
subplot(2,2,1)
plotCellData(model.G, state.N.c);view(30,60);
colorbar
title('N Concentration after 3 Timesteps')

% loop through all the time steps
for istate = 1 : numel(states)

    state = states{istate};

    % plotting un-reacted cells
    set(0, 'currentfigure', 1);
    subplot(2,2,1)
    plotCellData(model.G, state.R.c);view(30,60);
    colorbar
    title('R Concentration')
    
    % plotting neurotransmitters
    set(0, 'currentfigure', 2);
    cla
    plotCellData(model.G, state.N.c); view(30,60);
    colorbar
    title('N Concentration')

    % plotting reacted cells
    set(0, 'currentfigure', 3);
    cla
    plotCellData(model.G, state.R_N.c);view(30,60);
    colorbar
    title('R-N Concentration')

    % counting number of reacted and un-reacted
    C_R_vec(istate) = sum(state.R.c);
    C_RN_vec(istate) = sum(state.R_N.c);

    set(0,'currentfigure',4);
    cla
    plot(1:n,C_R_vec, "B", 1:n, C_RN_vec, "R");
    title("Concentration of Receptors")
    legend("C_{R}","C_{RN}")
    
    drawnow
    pause(0.0001)
end

% displaying time took to react more than half the reactor cells.
transmitt = find(C_R_vec < C_RN_vec,1,"first");
disp(["signal transmitted at timestep ", transmitt]);
disp(["time to transmitt", transmitt*dt]);