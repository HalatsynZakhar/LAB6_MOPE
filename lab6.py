from math import sqrt
from random import randint
import numpy as np
import copy
import scipy.stats


def r(x: float) -> float:
    """Точність округлення"""
    x = round(x, 2)
    if float(x) == int(x):
        return int(x)
    else:
        return x


def par(a: float) -> str:
    """Для вивіду. Негативні числа закидає в скобки и округлює"""
    if a < 0:
        return "(" + str(r(a)) + ")"
    else:
        return str(r(a))


def average(list: list, name: str) -> int or float:
    """Середнє значення з форматованним вивідом для будь-якого листа"""
    print("{} = ( ".format(name), end="")
    average = 0
    for i in range(len(list)):
        average += list[i]
        if i == 0:
            print(r(list[i]), end="")
        else:
            print(" + ", end="")
            print(par(list[i]), end="")
    average /= len(list)
    print(" ) / {} = {} ".format(len(list), r(average)))
    return average


def printf(name: str, value: int or float):
    """Форматованний вивід змінної з округленням"""
    print("{} = {}".format(name, r(value)))


def matrixplan(xn_factor: list, x_min: int, x_max: int) -> list:
    """Заповнює матрицю планування згідно нормованої"""
    xn_factor_experiment = []
    for i in range(len(xn_factor)):
        if xn_factor[i] == -1:
            xn_factor_experiment.append(x_min)
        elif xn_factor[i] == 1:
            xn_factor_experiment.append(x_max)
    return xn_factor_experiment


def MatrixExper(x_norm: list, x_min: list, x_max: list) -> list:
    """Генеруємо матрицю планування згідно нормованної"""
    x_factor_experiment = []
    for i in range(len(x_norm)):
        x_factor_experiment.append(matrixplan(x_norm[i], x_min[i], x_max[i]))
    return x_factor_experiment


def func_Y(xn_factor_experiment, m):
    """функція за варіантом. Повертає подвійний лист з Y"""
    list = []
    for j in range(m):
        list.append([])
        for i in range(len(xn_factor_experiment[0])):
            x1 = xn_factor_experiment[0][i]
            x2 = xn_factor_experiment[1][i]
            x3 = xn_factor_experiment[2][i]
            list[j].append(
                9.9 + 2.7 * x1 + 0.3 * x2 + 1.4 * x3 + 6.8 * x1 ** 2 + 0.6 * x2 ** 2 + 3.9 * x3 ** 2 + 1.6 * x1 * x2 +
                0.1 * x1 * x3 + 1.7 * x2 * x3 + 9.9 * x1 * x2 * x3 + randint(0, 10) - 5)
    return list


def a(x_i_list: list, col1: int, col2: int):
    """рахуэ a mn. Кожный элемент одного стовпця множить на кожен элемент другого. Ділить на кількість доданків
    Необхідно для знаходження коефіціентів"""
    temp_list = []
    for i in range(len(x_i_list[0])):
        temp_list.append(x_i_list[col1][i] * x_i_list[col2][i])
    sum_res = sum(temp_list) / len(temp_list)
    return sum_res


def dispers(y: list, y_average_list: list, m) -> list:
    """Рахує s2 для усіх рядків. Повертає масив значень
    Необхідно для критерію Кохрена"""
    s2_y_row = []

    for i in range(len(y_average_list)):
        s2_y_row.append(0)
        print("s2_y_row{} = ( ".format(i + 1), end="")
        for j in range(m):
            s2_y_row[i] += (y[j][i] - y_average_list[i]) ** 2
            if j == 0:
                print("({} - {})^2".format(r(y[j][i]), par(y_average_list[i])), end="")
            else:
                print(" + ({} - {})^2".format(r(y[j][i]), par(y_average_list[i])), end="")
        s2_y_row[i] /= m
        print(" ) / {} = {} ".format(m, r(s2_y_row[i])))

    return s2_y_row


def beta(x_norm: list, y_average_list: list) -> list:
    """Рахує Бета критерия Стюдента. Повертає масив значень"""
    beta_list = []

    for i in range(len(x_norm)):
        beta_list.append(0)
        print("Beta{} = ( ".format(i + 1), end="")
        for j in range(len(x_norm[i])):
            beta_list[i] += y_average_list[j] * x_norm[i][j]
            if j == 0:
                print("{}*{}".format(r(y_average_list[j]), par(x_norm[i][j])), end="")
            else:
                print(" + {}*{}".format(r(y_average_list[j]), par(x_norm[i][j])), end="")
        beta_list[i] /= len(x_norm[0])
        print(" ) / {} = {} ".format(len(x_norm[0]), r(beta_list[i])))

    return beta_list


def t(beta_list: list, s_BetaS) -> list:
    """Рахує t критерія Стюдента. Повертає масив значень"""
    t_list = []
    for i in range(len(beta_list)):
        t_list.append(abs(beta_list[i]) / s_BetaS)
        print("t{} = {}/{} = {}".format(i, r(abs(beta_list[i])), par(s_BetaS), par(t_list[i])))
    return t_list


def s2_od_func(y_average_list, y_average_row_Student, m, N, d):
    """Вираховує сігму в квадраті для критерія Фішера"""
    s2_od = 0
    print("s2_od = ( ", end="")
    for i in range(len(y_average_list)):
        s2_od += (y_average_row_Student[i] - y_average_list[i]) ** 2
        if i == 0:
            print("({} - {})^2".format(r(y_average_row_Student[i]), par(y_average_list[i])), end="")
        else:
            print(" + ({} - {})^2".format(r(y_average_row_Student[i]), par(y_average_list[i])), end="")
    s2_od *= m / (N - d)
    print(" ) * {}/({} - {}) = {} ".format(m, N, d, r(s2_od)))
    return s2_od


x_min = [10, -15, -15]  # Задані за умовою значення. Варіант 206
x_max = [40, 35, 5]

x_average_min = average(x_min, "X_average_min")  # Середнє Х макс и мин
x_average_max = average(x_max, "X_average_max")

m = 3  # За замовчуванням
q = 0.05  # рівень значимості

x_0_i = [(x_min[i] + x_max[i]) / 2 for i in range(len(x_min))]  # x01, x02, x03
delta_x_i = [(x_max[i] - x_0_i[i]) for i in range(len(x_min))]  # delta_x1, delta_x2, delta_x3

x_norm = [[-1, -1, -1, -1, +1, +1, +1, +1],  # нормована матриця
          [-1, -1, +1, +1, -1, -1, +1, +1],
          [-1, +1, -1, +1, -1, +1, -1, +1]]

x_factor_experiment = MatrixExper(x_norm, x_min, x_max)  # генеруємо експер. матрицю згідно нормованої

x_norm.insert(0, [+1, +1, +1, +1, +1, +1, +1, +1])  # додаємо перший стовпчик з одиницям в нульову позицію

l = sqrt(3)  # відстань до зоряної точки
x_norm[0].extend([1 for i in range(6)])  # x0 в норм. матриці з одиниць
x_norm[1].extend([l, -l, 0, 0, 0, 0])  # x1_norm
x_norm[2].extend([0, 0, l, -l, 0, 0])  # x2_norm
x_norm[3].extend([0, 0, 0, 0, l, -l])  # x3_norm

x_norm.append([x_norm[1][i] * x_norm[2][i] for i in range(len(x_norm[0]))])  # додаємо ефект взаимодії і квадрат. x12
x_norm.append([x_norm[1][i] * x_norm[3][i] for i in range(len(x_norm[0]))])  # x13
x_norm.append([x_norm[2][i] * x_norm[3][i] for i in range(len(x_norm[0]))])  # x23
x_norm.append([x_norm[1][i] * x_norm[2][i] * x_norm[3][i] for i in range(len(x_norm[0]))])  # x12
x_norm.append([x_norm[1][i] ** 2 for i in range(len(x_norm[0]))])  # x1^2
x_norm.append([x_norm[2][i] ** 2 for i in range(len(x_norm[0]))])  # x2^2
x_norm.append([x_norm[3][i] ** 2 for i in range(len(x_norm[0]))])  # x3^2

"""Додаємо зоряні точки до експерементальної матриці"""
x_factor_experiment[0].extend(
    [l * delta_x_i[0] + x_0_i[0], -l * delta_x_i[0] + x_0_i[0], x_0_i[0], x_0_i[0], x_0_i[0], x_0_i[0]])
x_factor_experiment[1].extend(
    [x_0_i[1], x_0_i[1], l * delta_x_i[1] + x_0_i[1], -l * delta_x_i[1] + x_0_i[1], x_0_i[1], x_0_i[1]])
x_factor_experiment[2].extend(
    [x_0_i[2], x_0_i[2], x_0_i[2], x_0_i[2], l * delta_x_i[2] + x_0_i[2], -l * delta_x_i[2] + x_0_i[2]])

N = len(x_factor_experiment[0])  # кількість рядків
count = 0  # лічильник кількості спроб, щоб задовільнити всі критерії
while True:  # Вихід тільки якщо задовольняються критерії
    count += 1  # лічильник спроб

    y = func_Y(x_factor_experiment, m)  # генеруємо значення функції відгуку)

    y_average_list = []  # cереднє значення рядка Y
    for i in range(len(y[0])):
        y_average_list.append(
            average([y[j][i] for j in range(m)], "y_average_{}row".format(i + 1)))  # рахую середнє У у рядках

    y_average_average = average(y_average_list, "Y_average_average")  # середнє середніх значень Y

    x_exper_neight = []  # урахування еферкту взаємодії і квадратних членів. В массиві стовпці.
    for j in range(7):
        x_exper_neight.append([])
        for i in range(len(x_factor_experiment[0])):
            if j == 0:
                x_exper_neight[j].append(x_factor_experiment[0][i] * x_factor_experiment[1][i])  # x1x2
            if j == 1:
                x_exper_neight[j].append(x_factor_experiment[0][i] * x_factor_experiment[2][i])  # x1x3
            if j == 2:
                x_exper_neight[j].append(x_factor_experiment[1][i] * x_factor_experiment[2][i])  # x2x3
            if j == 3:
                x_exper_neight[j].append(
                    x_factor_experiment[0][i] * x_factor_experiment[1][i] * x_factor_experiment[2][i])  # x1x2x3
            if j == 4:
                x_exper_neight[j].append(x_factor_experiment[0][i] ** 2)  # x1**2
            if j == 5:
                x_exper_neight[j].append(x_factor_experiment[1][i] ** 2)  # x2**2
            if j == 6:
                x_exper_neight[j].append(x_factor_experiment[2][i] ** 2)  # x3**2

    x_i_list = []  # об ’эднанний лист факторів і ефекту взяємодії
    x_i_list.extend(x_factor_experiment)
    x_i_list.extend(x_exper_neight)

    list_mx = []  # середнє по усім стовпцям
    for i in range(len(x_i_list)):
        list_mx.append(average(x_i_list[i], "m{}".format(i + 1)))

    my = y_average_average  # cереднє всіх значень функції

    """Генерація таблиці зі зручним виводом"""
    name = ["N", "x1", "x2", "x3", "x12", "x23", "x13", "x123", "x1^2", "x2^2", "x3^2"]
    name.extend(["y{}".format(i + 1) for i in range(m)])
    name.append("y_mid")

    for j in range(12 + m):
        print("|{: ^13}|".format(name[j]), end="")
    print()
    for i in range(12 + m):
        print("|{: ^13}|".format("---------"), end="")
    print()
    for i in range(N):
        print("|{: ^13}|".format(i + 1), end="")
        for j in range(10):
            print("|{: ^13}|".format(r(x_i_list[j][i])), end="")
        for j in range(m):
            print("|{: ^13}|".format(r(y[j][i])), end="")
        print("|{: ^13}|".format(r(y_average_list[i])), end="")
        print()
    for i in range(12 + m):
        print("|{: ^13}|".format("---------"), end="")
    print()

    print("|{: ^13}|".format("Mid"), end="")
    for i in list_mx:
        print("|{: ^13}|".format(r(i)), end="")
    for i in y:
        print("|{: ^13}|".format(r(sum(i) / len(i))), end="")
    print("|{: ^13}|".format(r(y_average_average)), end="")
    print()
    """Кінець генерації таблиці"""

    a_n = []  # a1, a2, a3......., a10
    for i in range(len(x_i_list)):
        temp = 0
        for j in range(N):
            temp += x_i_list[i][j] * y_average_list[j]
        a_n.append(temp / N)

    for i in range(len(a_n)):
        print("a_{} = {}".format(i + 1, r(a_n[i])), end="\t")
    print()

    a_mn = []  # a11, a12, a12...., a10 10
    for i in range(len(x_i_list)):
        a_mn.append([])
        for j in range(len(x_i_list)):
            a_mn[i].append("")
            a_mn[i][j] = a(x_i_list, i, j)

    print()
    for i in range(len(a_mn)):
        for j in range(len(a_mn)):
            print("a{: ^2}:{: ^2} = |{: ^13}|".format(j + 1, i + 1, par(a_mn[i][j])), end="\t")
        print()

    line1_10 = []  # знаменник для визначення коеф.

    line1_10.append([14])  # Заповнюю першу строку (інші строки - циклом)
    line1_10[0].extend([list_mx[i] for i in range(10)])

    for i in range(10):  # заповнення інших рядків матриці
        line1_10.append([list_mx[i]])
        for j in range(10):
            line1_10[i + 1].append(a_mn[j][i])

    vilni = [my]  # стовпець вільних членів
    vilni.extend([a_n[i] for i in range(10)])

    numer = [copy.deepcopy(line1_10) for i in range(11)]  # містить зараз 11 копій знаменника
    for i in range(11):  # доступаємось до кожного нумератора
        for j in range(11):
            numer[i][j][i] = vilni[j]  # міняє потрібний стовпець. Тепер у массиві нумер двувимірні массиви чисельників

    denominator = np.array(line1_10)  # знаменник
    numerator = []  # масив, що зберігає чисельники (двовимірні массиви)
    for i in range(len(numer)):
        numerator.append(np.array(numer[i]))
    print("\nЗнаменник")
    for i in denominator:
        for j in i:
            print("|{: ^13}|".format(round(j, 1)), end="")
        print()

    for k in range(len(numerator)):
        print("\nЧисельник {}".format(k + 1))
        for i in numerator[k]:
            for j in i:
                print("|{: ^13}|".format(round(j, 1)), end="")
            print()

    try:
        b = []
        for i in range(len(numerator)):
            print(np.linalg.det(numerator[i]), end="")
            print(" /  ", end="")
            print(np.linalg.det(denominator), end="")
            print(" = ", end="")
            b.append(np.linalg.det(numerator[i]) / np.linalg.det(denominator))
            print(par(b[i]))
    except:
        print("Невизначена помилка під час роботи з матрицями")
        break

    print(
        "\ny = {} + {}*x1 + {}*x2 + {}*x3 + {}*x1*x2 + {}*x1*x3 + {}*x2*x3 + {}*x1*x2*x3 + {}*x1^2 + {}*x2*2 + {}*x3^2\n"
            .format(par(b[0]), par(b[1]), par(b[2]), par(b[3]), par(b[4]), par(b[5]), par(b[6]), par(b[7]), par(b[8]),
                    par(b[9]), par(b[10])))

    y_control = []  # Розраховуємо утворені значення для перевірки з середніми
    for i in range(len(x_i_list[0])):
        temp = 0
        temp = b[0] + b[1] * x_i_list[0][i] + b[2] * x_i_list[1][i] + b[3] * x_i_list[2][i] + b[4] * x_i_list[3][i] \
               + b[5] * x_i_list[4][i] + b[6] * x_i_list[5][i] + b[7] * x_i_list[6][i] \
               + b[8] * x_i_list[7][i] + b[9] * x_i_list[8][i] + b[10] * x_i_list[9][i]
        y_control.append(temp)

    for i in range(len(y_control)):
        print("Отримане значення: {}.\t\t\tСередне значення рядка Y: {}".format(r(y_control[i]), r(y_average_list[i])))
    print()

    print("\nКритерій Кохрена\n")
    s2_list = dispers(y, y_average_list, m)  # дисперсії по рядках

    Gp = max(s2_list) / sum(s2_list)
    print("Gp = (max(s2) / sum(s2)) = {}".format(par(Gp)))
    print("f1=m-1={} ; f2=N=4 Рівень значимості приймемо 0.05.".format(m))
    f1 = m - 1
    f2 = N
    p = 0.95
    q = 1 - p
    Gt_tableN12 = {1: 0.5410, 2: 0.3924, 3: 0.3264, 4: 0.2880, 5: 0.2624, 6: 0.2439, 7: 0.2299, 8: 0.2187, 9: 0.2098,
                   10: 0.2020, 16: 0.1737, 36: 0.1403, 144: 0.0100, "inf": 0.0833}  # f2 = 12, рівень знач. 0.05
    if f1 <= 10:
        Gt = Gt_tableN12[f1]  # табличне значення критерію Кохрена при N=4, f1=2, рівень значимості 0.05
    elif f1 <= 16:
        Gt = Gt_tableN12[16]
    elif f1 <= 36:
        Gt = Gt_tableN12[36]
    elif f1 <= 144:
        Gt = Gt_tableN12[144]
    else:
        Gt = Gt_tableN12["inf"]
    printf("Gt", Gt)
    if Gp <= Gt:
        Krit_Kohr = "Однор" + " m=" + str(m)
        print("Дисперсія однорідна")

    else:
        Krit_Kohr = "Не однор."
        print("Дисперсія неоднорідна\n\n\n\n")
        print("m+1")
        m += 1

        continue  # цикл починається знову, якщо неоднор. Якщо однорідне, то цикл продовжується

    print("\nКритерію Стьюдента\n")

    """Генерація таблиці зі зручним виводом"""
    print("Нормовані фактори")
    name = ["N", "x0", "x1", "x2", "x3", "x12", "x23", "x13", "x123", "x1^2", "x2^2", "x3^2"]
    name.extend(["y{}".format(i + 1) for i in range(m)])
    name.append("y_mid")

    for j in range(13 + m):
        print("|{: ^13}|".format(name[j]), end="")
    print()
    for i in range(13 + m):
        print("|{: ^13}|".format("---------"), end="")
    print()
    for i in range(N):
        print("|{: ^13}|".format(i + 1), end="")
        for j in range(11):
            print("|{: ^13}|".format(r(x_norm[j][i])), end="")
        for j in range(m):
            print("|{: ^13}|".format(r(y[j][i])), end="")
        print("|{: ^13}|".format(r(y_average_list[i])), end="")
        print()
    for i in range(13 + m):
        print("|{: ^13}|".format("---------"), end="")
    print()

    """Кінець генерації таблиці"""

    s2_B = sum(s2_list) / len(s2_list)
    printf("s2_B", s2_B)

    s2_BetaS = s2_B / (N * m)
    printf("s2_BetaS", s2_BetaS)

    s_BetaS = sqrt(s2_BetaS)
    printf("s_betaS", s_BetaS)

    beta_list = beta(x_norm, y_average_list)  # значенння B0, B1, B2, B3....

    t_list = t(beta_list, s_BetaS)  # t0, t1, t2, t3

    f3 = (m - 1) * N  # N завжди 14
    t_tabl = scipy.stats.t.ppf((1 + (1 - q)) / 2, f3)  # табличне значення за критерієм Стюдента
    printf("t_tabl", t_tabl)

    b_list_St = []
    print("Утворене рівняння регресії: Y = ", end="")
    for i in range(len(t_list)):
        """Форматованне виведення рівняння зі значущими коеф. Не знач. пропускаються"""
        b_list_St.append(0)
        if t_list[i] > t_tabl:
            b_list_St[i] = b[i]
            if i == 0:
                print("{}".format(r(b[i])), end="")
            else:
                print(" + {}*{}".format(par(b[i]), name[i + 1]), end="")
    print()

    # Порівняння результатів
    y_average_row_Student = []
    dodanki = []  # для гарного виведення, буде зберігати текст
    for i in range(len(x_i_list[0])):
        for j in range(len(b_list_St)):
            if j == 0:
                dodanki.append("{}".format(r(b_list_St[j])))  # додає доданок до виведення, якщо він не нуль
            else:
                if b_list_St[j] == 0:
                    dodanki.append("")  # додає доданок до виведення як пустий, якщо він нуль
                else:
                    dodanki.append(" + {}*{}".format(par(b_list_St[j]), x_i_list[j - 1][i]))
        y_average_row_Student.append(0)
        y_average_row_Student[i] = b[0] + b[1] * x_i_list[0][i] + b[2] * x_i_list[1][i] + b[3] * x_i_list[2][i] + b[4] * \
                                   x_i_list[3][i] \
                                   + b[5] * x_i_list[4][i] + b[6] * x_i_list[5][i] + b[7] * x_i_list[6][i] \
                                   + b[8] * x_i_list[7][i] + b[9] * x_i_list[8][i] + b[10] * x_i_list[9][i]

        if abs(y_average_row_Student[i] - y_average_list[i]) >= 20:
            print("Yrow{} = {}{}{}{} = \033[31m {}\t\t\t\033[0mY_average_{}row = \033[31m {}\033[0m".format(
                i + 1, dodanki[0], dodanki[1], dodanki[2], dodanki[3],
                r(y_average_row_Student[i]), i + 1, r(y_average_list[i])))
        elif abs(y_average_row_Student[i] - y_average_list[i]) >= 10:
            print("Yrow{} = {}{}{}{} = {}\t\t\tY_average_{}row =  {}".format(
                i + 1, dodanki[0], dodanki[1], dodanki[2], dodanki[3],
                r(y_average_row_Student[i]), i + 1, r(y_average_list[i])))
            print("Результат приблизно (+-10) збігається! (Рівень значимості 0.05)")
        else:
            print("Yrow{} = {}{}{}{} = {}\t\t\tY_average_{}row =  {}".format(
                i + 1, dodanki[0], dodanki[1], dodanki[2], dodanki[3],
                r(y_average_row_Student[i]), i + 1, r(y_average_list[i])))
            print("Результат приблизно (+-10) збігається! (Рівень значимості 0.05)")
        dodanki.clear()

    print("Критерій Фішера")
    d = len(b_list_St) - b_list_St.count(0)
    f4 = N - d
    s2_od = s2_od_func(y_average_list, y_average_row_Student, m, N, d)

    Fp = s2_od / s2_B
    print("Fp = {} / {} = {}".format(r(s2_od), par(s2_B), r(Fp)))

    F_table = scipy.stats.f.ppf(1 - q, f4, f3)
    printf("F_table", F_table)

    if Fp > F_table:
        print("За критерієм Фішера рівняння регресії неадекватно оригіналу при рівні значимості 0.05")
        Krit_Fish = "Не адекв. "

        print("\nПочаток\n")

        continue  # знову зі збільшенням m

    else:
        print("За критерієм Фішера рівняння регресії адекватно оригіналу при рівні значимості 0.05")
        printf("Номер спроби:", count)
        Krit_Fish = "Адекв."

        break  # якщо программа дійшла до цієї точки, то все виконано вірно, критерії задовольняють умову
