classdef DiffusionComponentInputParams < ComponentInputParams

    properties
        
        D
        
    end
    
    
    methods

        function paramobj = DiffusionComponentInputParams(jsonstruct)
            paramobj = paramobj@ComponentInputParams(jsonstruct);
        end
        
    end
    

    
end
