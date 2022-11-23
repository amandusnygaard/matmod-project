classdef ReactionDiffusion < BaseModel

    properties
        
        R % Diffision model for component R
        N % Diffision model for component N
        R_N % Diffision model for result
        
        k_1 % reaction constant k_1
        k_2 % reaction constant k_2
        
    end
    
    methods
        
        function model = ReactionDiffusion(paramobj)
            
            model = model@BaseModel();
            
            % All the submodels should have same backend (this is not assigned automaticallly for the moment)
            model.AutoDiffBackend = SparseAutoDiffBackend('useBlocks', false);
            
            %% Setup the model using the input parameters
            fdnames = {'G'            , ...
                       'k_1'          , ...
                       'k_2'};
            
            model = dispatchParams(model, paramobj, fdnames);

            model.R = DiffusionComponent(paramobj.R);
            model.N = DiffusionComponent(paramobj.N);
            model.R_N = DiffusionComponent(paramobj.R_N);
            
        end

        
        function model = registerVarAndPropfuncNames(model)
            
            %% Declaration of the Dynamical Variables and Function of the model
            % (setup of varnameList and propertyFunctionList)
            
            model = registerVarAndPropfuncNames@BaseModel(model);
            
            %% Temperature dispatch functions
            fn = @ReactionDiffusion.updateSourceTerm;
            
            inputnames = {{'R', 'c'}, ...
                          {'N', 'c'}};
            model = model.registerPropFunction({{'R', 'source'} , fn, inputnames});
            model = model.registerPropFunction({{'N', 'source'} , fn, inputnames});
            model = model.registerPropFunction({{'R_N', 'source'} , fn, inputnames});
            
        end

        function forces = getValidDrivingForces(model)
            forces = getValidDrivingForces@PhysicalModel(model);
            forces.none = [];
        end
        
        function state = updateSourceTerm(model, state)

            k_1 = model.k_1;
            k_2 = model.k_2;
            vols = model.G.cells.volumes;
            
            cR = state.R.c;
            cN = state.N.c;
            cR_N = state.R_N.c;

            Re1 = k_1.*vols.*cR.*cN;
            Re2 = k_2.*vols.*cR_N;

            state.R.source = -Re1 +Re2;
            state.N.source = -Re1 +Re2;
            state.R_N.source = Re1 -Re2;
            
        end
        
        
        function [problem, state] = getEquations(model, state0, state, dt, drivingForces, varargin)
            
            state = model.initStateAD(state);

            state.R = model.R.updateFlux(state.R);
            state.N = model.N.updateFlux(state.N);
            state.R_N = model.R_N.updateFlux(state.R_N);
            
            state = model.updateSourceTerm(state);

            state.R = model.R.updateMassAccum(state.R, state0.R, dt);
            state.N = model.N.updateMassAccum(state.N, state0.N, dt);
            state.R_N = model.R_N.updateMassAccum(state.R_N, state0.R_N, dt);
            
            state.R = model.R.updateMassConservation(state.R);
            state.N = model.N.updateMassConservation(state.N);
            state.R_N = model.R_N.updateMassConservation(state.R_N);
            
            eqs = {}; types = {}; names = {};
            
            eqs{end + 1}   = state.R.massCons;
            names{end + 1} = 'massCons R';
            types{end + 1} = 'cell';
            
            eqs{end + 1}   = state.N.massCons;
            names{end + 1} = 'massCons N';
            types{end + 1} = 'cell';

            eqs{end + 1}   = state.R_N.massCons;
            names{end + 1} = 'massCons R_N';
            types{end + 1} = 'cell';
                        
            primaryVars = model.getPrimaryVariables();

            %% Setup LinearizedProblem that can be processed by MRST Newton API
            problem = LinearizedProblem(eqs, types, names, primaryVars, state, dt);
            
        end
        
        
        function state = initStateAD(model, state)
        % initialize a new cleaned-up state with AD variables

            % initStateAD in BaseModel erase all fields
            newstate = initStateAD@BaseModel(model, state);
            newstate.time = state.time;
            state = newstate;
            
        end 

        
        function primaryvarnames = getPrimaryVariables(model)

            primaryvarnames = {{'R', 'c'}, ...
                               {'N', 'c'}, ...
                               {'R_N', 'c'}};
            
        end
        

        function model = validateModel(model, varargin)
        % nothing special to do
        end


    end

end