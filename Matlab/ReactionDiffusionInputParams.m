
classdef ReactionDiffusionInputParams < InputParams

    properties
        
        G % the grid to solve over
        
        R % parameter for diffusion component model R
        N % parameter for diffusion component model N
        R_N % parameter for diffusion component model R_N
        
        k_1 % reaction rate k_1
        k_2 %reation rate k_{-1}

    end
    
    methods
        
        function paramobj = ReactionDiffusionInputParams(jsonstruct)
            
            paramobj = paramobj@InputParams(jsonstruct);
            
            pick = @(fd) pickField(jsonstruct, fd);
            
            paramobj.R = DiffusionComponentInputParams(pick('R'));
            paramobj.N = DiffusionComponentInputParams(pick('N'));
            paramobj.R_N = DiffusionComponentInputParams(pick('R_N'));

            paramobj = paramobj.validateInputParams();
            
        end

        function paramobj = validateInputParams(paramobj)

            if ~isempty(paramobj.G)
                paramobj.R.G = paramobj.G;
                paramobj.N.G = paramobj.G;
                paramobj.R_N.G = paramobj.G;
            end
            
        end
        
    end
    
end