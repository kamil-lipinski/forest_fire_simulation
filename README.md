Prezentacja działania ⬇️⬇️⬇️

[![Pożar lasu](https://img.youtube.com/vi/nsmnCwWA4ng/0.jpg)](https://www.youtube.com/watch?v=nsmnCwWA4ng)

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
