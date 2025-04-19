# Toy Training and Inference Engine (TTIE)

[![Build & Test](https://github.com/ckorikov/2025-ttie/actions/workflows/cmake-single-platform.yml/badge.svg)](https://github.com/ckorikov/2025-ttie/actions/workflows/cmake-single-platform.yml)

Совместный проект [курса в МФТИ](https://ckorikov.github.io/2025-spring-efficient-ai/) по методам эффективной реализации моделей искусственного интеллекта.


Цель проекта — реализовать библиотеку для обучения и инференса нейронной сети на CPU, GPU, NPU.

## Сборка

```bash
cmake -S . -B build
cmake --build build
```

### Запуск тестов

```bash
cd build
ctest
```

или 

```bash
cd build
./tests
```

### Запуск примера

```bash
cd build
./example
```

## Задачи

Вам нужно сделать 2 вклада в проект: добавить новую функцию и оптимизировать существующую.

### Функции

- [ ] Реализовать слои 
    - [ ] [Pooling слои](https://pytorch.org/docs/stable/nn.html#pooling-layers)
    - [ ] [`Conv1d`](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html), `Conv2d`, `Conv3d`
    - [ ] [`Softmax`](https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html#torch.nn.Softmax)
    - [ ] [`BatchNorm1d`](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html), `BatchNorm2d`, `BatchNorm3d`
    - [ ] [`LayerNorm`](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)
    - [ ] [`InstanceNorm1d`](https://pytorch.org/docs/stable/generated/torch.nn.InstanceNorm.html), `InstanceNorm2d`, `InstanceNorm3d`
    - [ ] [`RNN`](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)
    - [ ] [`LSTM`](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
    - [ ] [`GRU`](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html)
    - [ ] [`Multihead Attention`](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)
    - [ ] [Нелинейные функции активации](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity)
    - [ ] [`Bilinear`](https://pytorch.org/docs/stable/generated/torch.nn.Bilinear.html)
    - [ ] [Любые другие слои](https://pytorch.org/docs/stable/nn.html#)
- [Реализовать функции потерь](https://pytorch.org/docs/stable/nn.html#loss-functions)
- [Реализовать функции расстояния](https://pytorch.org/docs/stable/nn.html#distance-functions)

### Оптимизации

- [ ] [Алгоритмические методы ускорения matmul](https://ckorikov.github.io/2025-spring-efficient-ai/matmul_1.html#/)
- [ ] [Аппаратные методы ускорения matmul](https://ckorikov.github.io/2025-spring-efficient-ai/matmul_2.html#/)
- [ ] [Методы ускорения нейронных сетей](https://ckorikov.github.io/2025-spring-efficient-ai/dl_3.html#/6)
- [ ] Добавить параллельные вычисления в слой (`std::thread`, `OpenMP`, `CUDA`, `OpenСL`, другие)
- [ ] Gradient checkpointing
- [ ] [Operator fusion](https://medium.com/data-science/how-pytorch-2-0-accelerates-deep-learning-with-operator-fusion-and-cpu-gpu-code-generation-35132a85bd26#:~:text=What%20is%20operator,memory%20read/writes.)
- [ ] [Квантованый слой](https://habr.com/ru/companies/yandex/articles/800945/)
- [ ] Специфические для слоев методы оптимизаций. Например, для Softmax есть [оптимизация](https://habr.com/ru/companies/otus/articles/562918/), повышающая стабильность вычислений
- [ ] Сэкономить память, сохраняя только необходимые промежуточные активации (`std::vector<Tensor> activations`)

### Рефакторинг и другие улучшения

- [ ] [Скрыть](https://en.wikipedia.org/wiki/Encapsulation_(computer_programming)) внутреннюю струкутуру `Tensor` от пользователя
- [ ] Добавить классу `Tensor` возможность работать не только с `float`
- [ ] [Добавить](https://clang.llvm.org/docs/ClangFormat.html) автоматическое форматирование кода
- [ ] [Добавить](https://clang.llvm.org/extra/clang-tidy/) автоматическую проверку кода статическими анализаторами
- [ ] Интегрировать функциональную часть из main.cpp в библиотеку
- [ ] Изменить схему владения объектов класса `Layer`. Используя c++11 [можно](https://habr.com/ru/companies/piter/articles/706866/) сделать код более читаемым и безопасным
- [ ] Добавить в `CMakeLists.txt` гибкую конфигурацию проекта. Например, при дебаге имеет смысл собирать тесты, компилировать с флагом `-O0` и санитайзерами, а в релизной версии для ускорения работы программы этого делать не стоит
- [ ] Поработать с [CI/CD](https://habr.com/ru/companies/otus/articles/515078/). Можно [добавить](https://habr.com/ru/articles/252101/) автоматическую генерацию документации и [развертывание](https://habr.com/ru/articles/799051/) ее на [github pages](https://pages.github.com/). Можно добавить [автоматический](https://github.com/marketplace/actions/jest-coverage-report?ysclid=m9o4x1pbpn743679522) отчет о проценте [покрытия](https://gcovr.com/en/5.0/guide.html) тестами кода через [github actions](https://docs.github.com/en/actions)
- [ ] Любые оптимизации на уровне C++, экономящие память, ускоряющие время работы, или делающие код более безопасным или расширяемым
- [ ] Нахождение багов/undefined behavior в программе

## Комментарии 
- Представленные выше пункты списка неравнозначны: какие-то требуют глубокого понимания и большого числа строчек кода, например, написание блока механизма внимания `MultiheadAttention`. Какие-то задачи требуют только изменений в паре строк кода. Более сложные задачи = гарантия высокого балла и  получение ценного опыта. Оценивается уровень понимания и вклад в проект, поэтому можете брать несколько задач.
- Вопросы по задачам/предложение своих задач/вопосы по теории приветствуются. Задавайте их в чате курса или индивидуально [@ckorikov](https://t.me/ckorikov), [@Harper567](https://t.me/Harper567), [@GoshaSerbin](https://t.me/GoshaSerbin)


## Общие рекомендации при выполнении задания
- При возможности использовать современный стандарт C++ (хотя бы C++11) вместо чистого C
- При добавлении функционального кода необходимо убедиться, что он имеет ожидаемое поведение, написав тесты
- Вайб-кодинг допускается, но с пониманием что и зачем было сделано
- [Читаемые](https://www.conventionalcommits.org/en/v1.0.0/) названия коммитов
- В описании пул-реквеста напишите какие задачи выполнили/как работает ваша оптимизация
- При создании пул-реквеста указывайте меня [@GoshaSerbin](https://t.me/GoshaSerbin) в качестве reviewer и assignee, кидайте ссылку на реквест в личные сообщения, чтобы я точно увидел. Если замечаний нет я вливаю изменения в репозиторий, если есть - пишу их и отмечаю вас assignee. После исправления замечаний в том же реквесте укажите меня снова assignee и снова киньте ссылку. Процесс продолжается итеративно пока замечаний не останется
