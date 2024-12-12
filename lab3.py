import numpy as np
import json
import heapq

zadanie = 3

if zadanie == 1:
    print('Zadanie 1')
    matrix1 =   [[1 ,  1  , 1],
                [5  ,   6 , 1]]

    matrix2 =   [[1 ,  2    ,1 ],
                [5  ,   6   ,1 ]]



    def multiply(m1:list, m2:list):
        if len(m1[0]) != len(m2):
            print("matrix size error!")
            return np.nan
        rows_a = len(m1)
        columns_a = len(m1[0])
        columns_b = len(m2[0])

        result = [[0 for _ in range(columns_b)] for _ in range(rows_a)]

        for i in range(rows_a):
            for j in range(columns_b):
                for k in range(columns_a):
                    result[i][j] += m1[i][k] * m2[k][j]

        return result

    print(multiply(matrix1,matrix2))

if zadanie == 2:
    print('Zadanie 2')

    class PersonalData:
        def __init__(self,name,age, postalcode):
            self.name = name
            self.age = age
            self.postalcode = postalcode
        def exportToJSON(self):
            json_dict = {'Name': self.name, 'Age':  self.age, 'PostalCode': self.postalcode}
            with open("sample.json", "w") as outfile:
                json.dump(json_dict, outfile)
        def importJSON(self):
            with open('sample.json', 'r') as openfile:
                json_dict = json.load(openfile)
            self.name = json_dict['Name']
            self.age = json_dict['Age']
            self.postalcode = json_dict['PostalCode']
        def __str__(self):
            return  str('Name: ' + self.name + ' Age: ' + str(self.age) +  ' PostalCode: ' + str(self.postalcode))

    ja = PersonalData("Mateusz",22,44100)
    ja.exportToJSON()
    ja = PersonalData("Bartek", 21, 44110)
    print(ja)
    ja.importJSON()
    print(ja)

if zadanie == 3:
    print('Zadanie 3')


    class Graph:
        def __init__(self):
            self.graph = {}

        def add_node(self, node):
            if node not in self.graph:
                self.graph[node] = []

        def add_edge(self, from_node, to_node, weight):
            if from_node not in self.graph:
                self.add_node(from_node)
            if to_node not in self.graph:
                self.add_node(to_node)
            self.graph[from_node].append((to_node, weight))


    def Dijkstra(graph, start):
        distances = {node: float('inf') for node in graph.graph}
        distances[start] = 0
        predecessors = {node: None for node in graph.graph}
        priority_queue = [(0, start)]

        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)

            if current_distance > distances[current_node]:
                continue

            for neighbor, weight in graph.graph[current_node]:
                distance = current_distance + weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    predecessors[neighbor] = current_node
                    heapq.heappush(priority_queue, (distance, neighbor))

        return distances, predecessors


    if __name__ == "__main__":
        graph = Graph()

        graph.add_edge('A', 'B', 1)
        graph.add_edge('A', 'C', 4)
        graph.add_edge('B', 'C', 2)
        graph.add_edge('B', 'D', 6)
        graph.add_edge('C', 'D', 3)

        start_node = 'A'
        distances, predecessors = Dijkstra(graph, start_node)

        print("Najkrótsze dystanse:", distances)
        print("Poprzednicy:", predecessors)

if zadanie == 4:
    print('Zadanie 4')
    class TrieNode:
        def __init__(self):
            self.children = {}
            self.fail = None
            self.output = []


    class AhoCorasick:
        def __init__(self):
            self.root = TrieNode()

        def add_pattern(self, pattern):
            node = self.root
            for char in pattern:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.output.append(pattern)

        def build(self):
            from collections import deque
            queue = deque()
            for child in self.root.children.values():
                child.fail = self.root
                queue.append(child)

            while queue:
                current_node = queue.popleft()

                for char, child in current_node.children.items():
                    queue.append(child)
                    fail_node = current_node.fail
                    while fail_node is not None and char not in fail_node.children:
                        fail_node = fail_node.fail
                    child.fail = fail_node.children[char] if fail_node else self.root
                    child.output += child.fail.output if child.fail else []

        def search(self, text):
            node = self.root
            matches = []

            for i, char in enumerate(text):
                while node is not None and char not in node.children:
                    node = node.fail

                if node is None:
                    node = self.root
                    continue
                node = node.children[char]

                if node.output:
                    for pattern in node.output:
                        matches.append((i - len(pattern) + 1, pattern))

            return matches

    if __name__ == "__main__":
        aho = AhoCorasick()

        words = ["p", "P", "pierog", "maki", "5"]

        for word in words:
            aho.add_pattern(word)

        aho.build()

        text = "Przepis na pierogi ruskie, 1kg maki, 2 jajka, 5kg ziemniakow"
        print(aho.search(text))

if zadanie == 5:
    print('Zadanie 5')
    class State:
        def __init__(self, name, output):
            self.name = name
            self.output = output
            self.transitions = {}

        def add_transition(self, input_value, next_state):
            self.transitions[input_value] = next_state

        def get_next_state(self, input_value):
            return self.transitions.get(input_value, None)


    class Moore:
        def __init__(self, initial_state):
            self.current_state = initial_state

        def process_input(self, input_value):
            next_state = self.current_state.get_next_state(input_value)
            if next_state:
                self.current_state = next_state

        def get_output(self):
            return self.current_state.output


    if __name__ == "__main__":
        s1 = State("S1", 0)
        s2 = State("S2", 1)
        s3 = State("S3", 0)

        s1.add_transition(0, s2)
        s1.add_transition(1, s3)
        s2.add_transition(1, s2)
        s2.add_transition(0, s1)
        s3.add_transition(0, s2)
        s3.add_transition(1, s1)

        machine = Moore(s1)
        print("Sekwencja wyjściowa:")
        input_sequence = [0, 1, 0, 1, 1]
        for input_value in input_sequence:
            machine.process_input(input_value)
            print(machine.get_output())



if zadanie == 6:
    print('Zadanie 6')

    class UppercaseDecorator:
        def __init__(self, func):
            self.func = func

        def __call__(self, *args, **kwargs):
            result = self.func(*args, **kwargs)
            if isinstance(result, str):
                return result.upper()
            return result


    @UppercaseDecorator
    def Pierogi():
        return "Pierogi!"


    print(Pierogi())

if zadanie == 7:
        print('Zadanie 7')

        from dataclasses import dataclass
        import json

        @dataclass
        class PersonalData:
            name: str
            age: int
            postalcode: int

            def exportToJSON(self, file_path: str):
                with open(file_path, 'w') as outfile:
                    json.dump(self.__dict__, outfile)

            def importJSON(self, file_path: str):
                with open(file_path, 'r') as openfile:
                    json_dict = json.load(openfile)
                self.name = json_dict['name']
                self.age = json_dict['age']
                self.postalcode = json_dict['postalcode']


        if __name__ == "__main__":
            ja = PersonalData("Mateusz", 22, 44100)
            ja.exportToJSON("person.json")
            ja = PersonalData("Bartek", 21, 44110)
            print(ja)
            ja.importJSON("person.json")
            print(ja)
