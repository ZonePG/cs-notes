# Project 0: 2048

Project0 实现 2048 游戏的核心逻辑。

在实验开始前，可以通过这份 [Google From quiz](https://docs.google.com/forms/d/e/1FAIpQLSeqyhGv2Fpa6HtUfWV4iR71f7pGW6TmRmvtH-X0FXq1KfvE7A/viewform) 了解游戏逻辑。

### public static boolean emptySpaceExists(Board b)

查看 board 上是否有空的 tile。

```java
/** Returns true if at least one space on the Board is empty.
  *  Empty spaces are stored as null.
  */
public static boolean emptySpaceExists(Board b) {
    // TODO: Fill in this function.
    for (Tile tile : b) {
        if (tile == null) {
            return true;
        }
    }
    return false;
}
```

### public static boolean maxTileExists(Board b)

查看 board 上的 tile 的值是否到了 MAX_PIECE(2048)，也就是赢得游戏。

```java
/**
  * Returns true if any tile is equal to the maximum valid value.
  * Maximum valid value is given by MAX_PIECE. Note that
  * given a Tile object t, we get its value with t.value().
  */
public static boolean maxTileExists(Board b) {
    // TODO: Fill in this function.
    for (Tile tile : b) {
        if (tile == null) {
            continue;
        }
        if (tile.value() == MAX_PIECE) {
            return true;
        }
    }
    return false;
}
```

### public static boolean atLeastOneMoveExists(Board b)

检查 board 上是否有可以移动的方向，包括两种情况：
- board 上有空的 tile
- 两个相邻的 tile 的值是相同的，这里通过`checkTileMoveExists`方法实现，比较相邻的 tile 的值，注意获取 tile 时是否合法。

```java
/**
  * Returns true if there are any valid moves on the board.
  * There are two ways that there can be valid moves:
  * 1. There is at least one empty space on the board.
  * 2. There are two adjacent tiles with the same value.
  */
public static boolean atLeastOneMoveExists(Board b) {
    // TODO: Fill in this function.
    for (int row = 0; row < b.size(); row++) {
        for (int col = 0; col < b.size(); col++) {
            if (b.tile(col, row) == null) {
                return true;
            }
            if (checkTileMoveExists(b, col, row)) {
                return true;
            }
        }
    }
    return false;
}

private static boolean checkTileMoveExists(Board b, int col, int row) {
    Tile tile = b.tile(col, row);
    if (checkTileValid(b, col + 1, row)) {
        if (tile.value() == b.tile(col + 1, row).value()) {
            return true;
        }
    }
    if (checkTileValid(b, col - 1, row)) {
        if (tile.value() == b.tile(col - 1, row).value()) {
            return true;
        }
    }
    if (checkTileValid(b, col, row + 1)) {
        if (tile.value() == b.tile(col, row + 1).value()) {
            return true;
        }
    }
    if (checkTileValid(b, col, row - 1)) {
        if (tile.value() == b.tile(col, row - 1).value()) {
            return true;
        }
    }
    return false;
}

private static Boolean checkTileValid(Board b, int col, int row) {
    // check bound is valid
    if (col < 0 || col > b.size() - 1 || row < 0 || row > b.size() - 1) {
        return false;
    }
    // then check is null or not
    if (b.tile(col, row) == null) {
        return false;
    }
    return true;
}
```

### public boolean tilt(Side side)

tile move 的逻辑在这个方法里面实现，可以通过这份 [Google From quiz](https://docs.google.com/forms/d/e/1FAIpQLSeWimFUFs4IleCPMQ1BK-8UV-a5ITYD93YGL6DbwZ3MOh60lw/viewform) 理清 tile move 的逻辑。

结合`TestUpOnly`测试，首先针对 North 方向单独考虑实现逻辑，如下：
- 对于每一列col，从 row = board.size() - 1 开始遍历 tile，并获取从 row 开始(不包括 row) 的下一个 tile (nextTile)。
- 如果 tile 为空，或者 tile 与 nextTile 的值相同，则表示可以调用 move，将 nextTile 移动到当前 tile 的位置。
    - 如果 move 返回 true，表示这是一个 merge，可以增加分数。
    - 如果 move 返回 false 的话，表示将 nextTile 移动到当前位置，并移动过来的 nextTile 可以等待合并，因此需要将 row++，避免 for 循环的 row-- 从而跳过当前位置的合并检查。

调用`board.setViewingPerspective(side)`就可以将任何方向都视为面向 North 方向，就不用重写其他方向的逻辑代码了。

```java
/** Tilt the board toward SIDE. Return true iff this changes the board.
  *
  * 1. If two Tile objects are adjacent in the direction of motion and have
  *    the same value, they are merged into one Tile of twice the original
  *    value and that new value is added to the score instance variable
  * 2. A tile that is the result of a merge will not merge again on that
  *    tilt. So each move, every tile will only ever be part of at most one
  *    merge (perhaps zero).
  * 3. When three adjacent tiles in the direction of motion have the same
  *    value, then the leading two tiles in the direction of motion merge,
  *    and the trailing tile does not.
  */
public boolean tilt(Side side) {
    boolean changed;
    changed = false;

    // TODO: Modify this.board (and perhaps this.score) to account
    // for the tilt to the Side SIDE. If the board changed, set the
    // changed local variable to true.
    board.setViewingPerspective(side);
    for (int col = 0; col < board.size(); col++) {
        for (int row = board.size() - 1; row >= 0; row--){
            Tile tile = board.tile(col, row);
            int nextRow = nextNonNullTileRow(col, row);
            if (nextRow == -1) {
                break;
            }
            Tile nextTile = board.tile(col, nextRow);
            if (tile == null || tile.value() == nextTile.value()) {
                changed = true;
                if (board.move(col, row, nextTile)) {
                    score += 2 * nextTile.value();
                } else {
                    row++;
                }
            }
        }
    }
    board.setViewingPerspective(Side.NORTH);

    checkGameOver();
    if (changed) {
        setChanged();
    }
    return changed;
}

private int nextNonNullTileRow(int col, int row) {
    for (int pos = row - 1; pos >= 0; pos--) {
        if (board.tile(col, pos) != null) {
            return pos;
        }
    }
    return -1;
}
```