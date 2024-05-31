import numpy as np

def load_map(filename):
    return np.load(filename)

class Node:
    def __init__(self, row=-1, col=-1):
        self.left = self
        self.right = self
        self.up = self
        self.down = self
        self.column = self
        self.row = row
        self.col = col

class ColumnNode(Node):
    def __init__(self, col=-1):
        super().__init__(row=-1, col=col)
        self.size = 0

class DancingLinks:
    def __init__(self, board_size):
        self.board_size = board_size
        self.header = ColumnNode()
        total_columns = 6 * board_size - 3  # Corrected total columns calculation
        self.columns = [ColumnNode(i) for i in range(total_columns)]
        
        self.header.right = self.columns[0]
        self.columns[0].left = self.header
        
        for i in range(1, total_columns):
            self.columns[i - 1].right = self.columns[i]
            self.columns[i].left = self.columns[i - 1]
        
        self.columns[-1].right = self.header
        self.header.left = self.columns[-1]
        
        for r in range(board_size):
            for c in range(board_size):
                nodes = [
                    Node(r, c),  # column constraint
                    Node(r, board_size + r),  # row constraint
                    Node(r, 2 * board_size + (r - c) + (board_size - 1)),  # main diagonal constraint
                    Node(r, 4 * board_size - 1 + (r + c))  # anti-diagonal constraint
                ]
                
                for node in nodes:
                    print(f"Node: ({node.row}, {node.col})")
                    col_node = self.columns[node.col]
                    col_node.size += 1
                    node.column = col_node
                    node.down = col_node
                    node.up = col_node.up
                    col_node.up.down = node
                    col_node.up = node
                    
                    if col_node.down == col_node:
                        col_node.down = node
                        node.up = col_node

                    if nodes[0] != node:
                        node.left = nodes[nodes.index(node) - 1]
                        nodes[nodes.index(node) - 1].right = node

                nodes[0].left = nodes[-1]
                nodes[-1].right = nodes[0]
                
    def cover(self, column):
        column.right.left = column.left
        column.left.right = column.right
        row_node = column.down
        while row_node != column:
            node = row_node.right
            while node != row_node:
                node.down.up = node.up
                node.up.down = node.down
                node.column.size -= 1
                node = node.right
            row_node = row_node.down
    
    def uncover(self, column):
        row_node = column.up
        while row_node != column:
            node = row_node.left
            while node != row_node:
                node.column.size += 1
                node.down.up = node
                node.up.down = node
                node = node.left
            row_node = row_node.up
        column.right.left = column
        column.left.right = column

    def search(self, k, solution):
        if self.header.right == self.header:
            print(f"Solution found: {solution}")
            return solution

        column = self.header.right
        node = column
        while node != self.header:
            if node.size < column.size:
                column = node
            node = node.right

        self.cover(column)

        row_node = column.down
        while row_node != column:
            solution.append((row_node.row, row_node.col))
            node = row_node.right
            while node != row_node:
                self.cover(node.column)
                node = node.right
            
            result = self.search(k + 1, solution)
            if result:
                return result
            
            row, col = solution.pop()
            row_node = self.columns[col]
            node = row_node.left
            while node != row_node:
                self.uncover(node.column)
                node = node.left
            row_node = row_node.down

        self.uncover(column)
        return None

def dancing_links(board):
    N = len(board)
    dlx = DancingLinks(N)
    solution = dlx.search(0, [])
    if solution:
        result = np.zeros((N, N), dtype=int)
        for row, col in solution:
            if col < N:
                result[row][col] = 1
        return result
    else:
        return np.zeros((N, N), dtype=int)

def main():
    board = load_map('n_queens_problem/boards/board_8_420ef5f3-3307-4ee4-84b8-e4d722a5bd07.npy')
    result = dancing_links(board)
    print(result)

if __name__ == '__main__':
    main()
