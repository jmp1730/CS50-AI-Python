import sys
from crossword import *
import itertools
import copy
class CrosswordCreator():
    def __init__(self, crossword):
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        for variable in self.domains:
            words_to_remove = set()
            for word in self.domains[variable]:
                if len(word) != variable.length:
                    words_to_remove.add(word)
            for word in words_to_remove:
                self.domains[variable].remove(word)

    def revise(self, x, y):
        revised = False
        overlap = self.crossword.overlaps[x, y]
        if overlap is not None:
            words_to_remove = set()
            for x_word in self.domains[x]:
                overlap_char = x_word[overlap[0]]
                corresponding_y_chars = {w[overlap[1]] for w in self.domains[y]}
                if overlap_char not in corresponding_y_chars:
                    words_to_remove.add(x_word)
                    revised = True
            for word in words_to_remove:
                self.domains[x].remove(word)

        return revised

    def ac3(self, arcs=None):
        if arcs is None:
            queue = list(itertools.product(self.crossword.variables, self.crossword.variables))
            queue = [arc for arc in queue if arc[0] != arc[1] and self.crossword.overlaps[arc[0], arc[1]] is not None]
        else:
            queue = arcs
        while queue:
            arc = queue.pop(0)
            x, y = arc[0], arc[1]
            if self.revise(x, y):
                if not self.domains[x]:
                    print("ending ac3")
                    return False
                for z in (self.crossword.neighbors(x) - {y}):
                    queue.append((z, x))
        return True

    def assignment_complete(self, assignment):
        if set(assignment.keys()) == self.crossword.variables and all(assignment.values()):
            return True
        else:
            return False
        
    def consistent(self, assignment):
        if len(set(assignment.values())) != len(set(assignment.keys())):
            return False
        if any(variable.length != len(word) for variable, word in assignment.items()):
            return False
        for variable, word in assignment.items():
            for neighbor in self.crossword.neighbors(variable).intersection(assignment.keys()):
                overlap = self.crossword.overlaps[variable, neighbor]
                if word[overlap[0]] != assignment[neighbor][overlap[1]]:
                    return False
        return True

    def order_domain_values(self, var, assignment):
        num_choices_eliminated = {word: 0 for word in self.domains[var]}
        neighbors = self.crossword.neighbors(var)
        for word_var in self.domains[var]:
            for neighbor in (neighbors - assignment.keys()):
                overlap = self.crossword.overlaps[var, neighbor]
                for word_n in self.domains[neighbor]:
                    if word_var[overlap[0]] != word_n[overlap[1]]:
                        num_choices_eliminated[word_var] += 1
        sorted_list = sorted(num_choices_eliminated.items(), key=lambda x:x[1])
        return  [x[0] for x in sorted_list]

    def select_unassigned_variable(self, assignment):
        unassigned_variables = self.crossword.variables - assignment.keys()
        num_remaining_values = {variable: len(self.domains[variable]) for variable in unassigned_variables}
        sorted_num_remaining_values = sorted(num_remaining_values.items(), key=lambda x: x[1])
        if len(sorted_num_remaining_values) == 1 or sorted_num_remaining_values[0][1] != sorted_num_remaining_values[1][1]:
            return sorted_num_remaining_values[0][0]
        else:
            num_degrees = {variable: len(self.crossword.neighbors(variable)) for variable in unassigned_variables}
            sorted_num_degrees = sorted(num_degrees.items(), key=lambda x: x[1], reverse=True)
            return sorted_num_degrees[0][0]

    def backtrack(self, assignment):
        if self.assignment_complete(assignment):
            return assignment
        var = self.select_unassigned_variable(assignment)
        for value in self.order_domain_values(var, assignment):
            test_assignment = copy.deepcopy(assignment)
            test_assignment[var] = value
            if self.consistent(test_assignment):
                assignment[var] = value
                result = self.backtrack(assignment)
                if result is not None:
                    return result
            assignment.pop(var, None)
        return None

def main():
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()