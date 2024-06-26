Automat komórkowy - pożar lasu

Stany:
1 - drzewo;
2 - płonące drzewo;
3 - spalone drzewo;
4 - woda;

Zasady ewolucji:
- drzewo staje się płonącym drzewem z prawdopodobieństwem p, jeśli ma w sąsiedztwie płonące drzewo
- płonące drzewo w następnej generacji staje się spalonym drzewem
- spalone drzewo odnawia się po k iteracjach
- samozapłon drzewa następuje z prawdopodobieństwem ps (odpowiednio małe)
- uwzględnić wodę, która stanowi barierę dla ognia
- uwzględnić wiatr zmieniający prawdopodobieństwa rozprzestrzeniania się pożaru w różnych kierunkach, kierunek powinien zmieniać się co kilka iteracji

===========================================================================

Cellular automaton - forest fire

States:
1 - tree;
2 - burning tree;
3 - burned tree;
4 - water;

Evolution rules:
- a tree becomes a burning tree with probability p if it has a burning tree in its neighborhood
- a burning tree becomes a burned tree in the next generation
- a burned tree regenerates after k iterations
- tree self-ignition occurs with probability ps (relatively low)
- consider water, which acts as a barrier to fire
- consider wind changing the probabilities of fire spread in different directions, the direction should change every few iterations

===========================================================================

- python 3.11
- numpy 1.26.2
- matplotlib 3.8.2

===========================================================================

![image](https://github.com/kamil-lipinski/forest_fire_simulation/assets/59886846/7fba228a-49a6-48b4-b809-35050a0928d8)
![image](https://github.com/kamil-lipinski/forest_fire_simulation/assets/59886846/f7ac23bd-897f-4aa5-892a-ce140756318a)
![image](https://github.com/kamil-lipinski/forest_fire_simulation/assets/59886846/e3e15348-b0d7-4b47-b637-c578af403f03)
![image](https://github.com/kamil-lipinski/forest_fire_simulation/assets/59886846/2ea1240c-1792-410b-9fee-13be3465dcf3)
