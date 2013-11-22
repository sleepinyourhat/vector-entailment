classdef Queue<handle
    %% Queue Class
    % 
    % properties:
    %
    %       link- doubly lined list for storing data
    % 
    % methods:
    % 
    %       Queue- constructor for Queue class
    % 
    %       enqueue(queue,x)- method for inserting new element x into the
    %       queue. x may be of any data type.
    % 
    %       x=dequeue(queue)- method for getting the next element x in the
    %       queue
    %  
    %       isempty- returns true if the queue is empty and false otherwise
    %
    % exemple:
    % 
    %       queue=Queue;
    % 
    %       queue.enqueue('12');
    % 
    %       n=queue.dequeue;
    
    properties(Access=private)
        link
    end
    
    methods
        function queue=Queue
            queue.link=DoublyLinkedList([]);
        end
        
        function enqueue(queue,x)
           if isa(x,'Node')
               insert(queue.link,x);
           else
               insert(queue.link,Node(x));
           end
           if isempty(queue.link.tail)
               queue.link.tail=queue.link.head;
           end
        end
        
        function x=dequeue(queue)
            if ~isempty(queue.link.tail)
                x=queue.link.tail;
                delete(queue.link,queue.link.tail);
                
            else 
                error('The queue is empty');
            end
            
        end
        
        % check whether this queue is empty
        function tf=isempty(queue)
            tf=isempty(queue.link.tail);
            
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

