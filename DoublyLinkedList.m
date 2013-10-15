classdef DoublyLinkedList<handle
    %
    %% Doubly link List
    %
    % properties:
    %
    %       head- the head of the list
    %
    %       tail- the tail of the list
    %
    % methods:
    %
    %       DoublyLinkedList(a)- Constructor for the doubly linked list, a
    %       is any numeric type or char.
    %       
    %       listSearch(dbll,k)- used for searching k key in dbll doubly
    %       linked list.
    %
    %       insert(dbll,x)- used for inputing x new node in the beginig of
    %       the list dbll
    %
    %       delete(dbll,x)- used for removing the x node from the douly
    %       linked list dbll. Two remove a node with specific value first
    %       use listSearch to find the right node and then use delete
    %       method for deletion of the node.
    %
    % example:
    %
    %       dll=DoublyLinkedList([1 2 3 4 5 6]);
    % 
    %       node=Node('10');
    % 
    %       insert(dll,node);
    % 
    %       n=listSearch(dll,'10');
    %
    %       delete(dll,n);
    % 
    %       dll.tail
    % 
    %       dll.head
    
    properties
        head=[];
        tail=[];
    end
    
    methods
        % constructor
        function dbll=DoublyLinkedList(a)
            if ~isempty(a)
                dbll.head=Node(a(1));
                dbll.tail=dbll.head;
           
                for i=2:length(a)
                    
                    insert(dbll,Node(a(i)));
                end
            end 
        end
        
        % search method
        function n=listSearch(dbll,k)
            x=dbll.head;
            while ~isempty(x) && all(x.data~=k)
                x=x.next;
            end
            if isempty(x)
                error('The key is not in the list');
            end
            n=x;
                
        end
        
        % insert new node method
        function insert(dbll,x)
            x.next=dbll.head;
            if ~isempty(dbll.head)
                dbll.head.previous=x;
            end
            dbll.head=x;
            x.previous=[];
        end
        
        % deletion method
        function delete(dbll,x)
            if ~isempty(x.previous)
                x.previous.next=x.next;
            else
                dbll.head=x.next;
            end

            if x==dbll.tail
               dbll.tail=x.previous; 
            end
        end
        
        
        % set/get functions
        function head=get.head(dbll)
            head=dbll.head;
        end
        
        function set.head(dbll,head)
            dbll.head=head;
        end
        
        function tail=get.tail(dbll)
            tail=dbll.tail;
        end
        
        function set.tail(dbll,tail)
            dbll.tail=tail;
        end
    end
    
end

