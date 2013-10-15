classdef Stack<handle
    %% Queue Class
    % 
    % properties:
    %
    %       link- doubly lined list for storing data
    % 
    % methods:
    % 
    %       Stack- constructor for Stack class
    % 
    %       push(stck,x)- method for inserting new element x into the
    %       stck. x may be of any data type.
    % 
    %       x=pop(stck)- method for getting the next element x in the
    %       stck
        %
    %       isempty(stck)- returns true if the stack is empty and false otherwise
    % 
    % exemple:
    % 
    %       stck=Stack;
    % 
    %       stck.push(1);
    % 
    %       n=stck.pop;
    % by Hanan Kavitz under BSD licence

    
    properties
        link
    end
    
    methods
        % constructor
        function stck=Stack
            stck.link=DoublyLinkedList([]);
        end
        
        % method for pushing new elements into the stack
        function push(stck,x)
            if isa(x,'Node')
               insert(stck.link,x);
           else
               insert(stck.link,Node(x));
           end
           if isempty(stck.link.tail)
               stck.link.tail=stck.link.head;
           end
        end
        
        % method for pulling the last entered element from the stack
        function x=pop(stck)
            if ~isempty(stck.link.head)
                x=stck.link.head;
                delete(stck.link,stck.link.head);
                stck
            else 
                error('The stack is empty');
            end
        end
        
        % check whether the stack is empty
        function tf=isempty(stck)
            tf=isempty(stck.link.head);
        end
        % get/set functions
        function set.link(queue,link)
            queue.link=link;
        end
        
        function link=get.link(queue)
            link=queue.link;
        end
    end
    
end

