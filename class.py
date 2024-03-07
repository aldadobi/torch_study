class Human:
    """ Super Class """
    def __init__(self):
        # instance 속성!!
        self.name  = '사람이름'
        self.age   = '나이'
        self.city  = '사는도시'

    def show(self):
        print('사람 클래스의 메소드입니다.')

class Student(Human):
    """ Child Class """

    def __init__(self,name):
        self.name = name
        super().__init__()
        

    def show_name(self):
        print('사람의 이름은 : ',self.name)

    def show_age(self):
        print('사람의 나이는 : ',self.age)
        
a = Student('james')
a.show()        # 메소드 상속 
a.show_age()    # 인스턴스 속성 상속
a.show_name()   # 자식노드에서 속성을 변경