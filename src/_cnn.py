from abc import abstractmethod


class AbstractCnn:
    def __init__(self,
                 n_classes,
                 input_shape=(224, 224, 3)):
        self.n_classes = n_classes
        self.input_shape = input_shape

    def init(self, *args, **kwargs):
        """
        Init callable method.
        :param args:
        :param kwargs:
        :return:
        """
        self._init(*args, **kwargs)

    @abstractmethod
    def _init(self, *args, **kwargs):
        """
        Init abstract method.
        :param args:
        :param kwargs:
        :return:
        """
        pass

    @abstractmethod
    def _build(self, *args, **kwargs):
        pass

    def _setter(self, attribute_value, keyword_parameter, keyword_parameter_name):
        """
        Set parameter from a child class of CNN model.
        :param attribute_value:
        :param keyword_parameter:
        :param keyword_parameter_name:
        :return:
        """
        if keyword_parameter is not None:
            keyword_parameter_name = "Subclass set {}".format(keyword_parameter_name)
            self._print_parameter_value(keyword_parameter_name, keyword_parameter)
            return keyword_parameter
        else:
            return attribute_value

    @staticmethod
    def _print_parameter_value(parameter_name, parameter_value):
        """
        Print couple of parameter name and value.
        :param parameter_name:
        :param parameter_value:
        :return:
        """
        string = "{} : {}".format(parameter_name, parameter_value)
        print('-' * len(string) + "\n{}".format(string) + '\n' + '-' * len(string))

    @staticmethod
    def _output_message(string_to_print):
        print('-' * len(string_to_print))
        print(string_to_print)
        print('-' * len(string_to_print))
