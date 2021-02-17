def OnlyA(A, B, C):
    return A

def OnlyB(A, B, C):
    return B

def OnlyC(A, B, C):
    return C


def BasicSum(A, B, C):
    return A + B + C
    # return B + C


def BasicProduct(A, B, C):
    return A * B * C
    # return B * C


def SumOfSquares(A, B, C):
    return A * A + B * B + C * C
    # return B * B + C * C


def ProductOfSquares(A, B, C):
    return A * A * B * B * C * C


def SquareOfSum(A, B, C):
    return (A + B + C) * (A + B + C)


def SquareOfSummedSquares(A, B, C):
    return (A * A + B * B + C * C) * (A * A + B * B + C * C)


def PairwiseProduct(A, B, C):
    return A * B + B * C + A * C


def SumPairwiseProduct(A, B, C):
    return A + B + C + A * B + B * C + A * C


def ProductAddition(A, B, C):
    return A + A * B + A * C

expressions = {}
expressions[OnlyA] = "$K_1$"
expressions[OnlyB] = "$K_2$"
expressions[OnlyC] = "$K_3$"
expressions[BasicSum] = "$K_1 + K_2 + K_3$"
expressions[BasicProduct] = "$K_1 * K_2 * K_3$"
expressions[SumOfSquares] = "$K_1^2 + K_2^2 + K_3^2$"
expressions[ProductOfSquares] = "$K_1^2 * K_2^2 * K_3^2$"
expressions[SquareOfSum] = "$(K_1+K_2+K_3)^2$"
expressions[SquareOfSummedSquares] = "$(K_1^2+K_2^2+K_3^2)^2$"
expressions[PairwiseProduct] = "$K_1 * K_2 + K_2 * K_3 + K_1 * K_3$"
expressions[SumPairwiseProduct] = "$K_1 + K_2 + K_3 + K_1 * K_2 + K_2 * K_3 + K_1 * K_3$"
expressions[ProductAddition] = "$K_1 + K_1 * K_2 + K_1 * K_3$"


basic_combinations = [OnlyA, OnlyB, OnlyC]
# combinations = [OnlyA, OnlyB, OnlyC]
# combinations = [BasicProduct, BasicSum, PairwiseProduct, SumOfSquares]
# combinations = [BasicSum, BasicProduct, SumOfSquares, SquareOfSum, SquareOfSummedSquares, PairwiseProduct, SumPairwiseProduct, ProductAddition]
# combinations = [BasicSum, BasicProduct, SumOfSquares, ProductOfSquares, SquareOfSum, SquareOfSummedSquares, PairwiseProduct, SumPairwiseProduct, ProductAddition]
combinations = [BasicSum, BasicProduct, SumOfSquares, SquareOfSum, PairwiseProduct, ProductAddition]
combinations = [ProductAddition]




