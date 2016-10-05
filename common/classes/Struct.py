# encoding: utf-8



class Struct:
    """
    Structure data type.

    Examples of use:

    A=cStruct()
    A.property1=value1
    A.property2=value2
    ...

    A=cStruct(property1=value1,property2=value2,...)
    """

    def __init__(self,**keywords):
        self._fields=[]
        for sKey in list(keywords.keys()):
            setattr(self,sKey,keywords[sKey])


    def get_str(self,sSeparator):
        sString = ''
        lsAttributes = list(vars(self).keys())

        for sAttribute in lsAttributes:
            if sAttribute!='_fields':
                sAttributeValue = str(getattr(self,sAttribute))
                sString+='%s=%s,%s'%(sAttribute,sAttributeValue,sSeparator)
        return '{'+sString[:-2]+'}'


    @property
    def pstr(self):
        return self.get_str('\n')

    def __str__(self):
        return self.get_str(' ')

    def __repr__(self):
        return self.get_str(' ')

    def __unicode__(self):
        return self.get_str(' ')

    def __eq__(self, other):
        return self.GetAttributes2ValuesDict()==other.GetAttributes2ValuesDict()

    def __ne__(self, other):
        return self.GetAttributes2ValuesDict()!=other.GetAttributes2ValuesDict()


    def __hash__(self):
        return hash(self.GetAttributes2ValuesTuple())


    def get_defaults(self,oDefaultStruct):
        '''
        Инициализация текущей структуры значениями по умолчанию из другой структуры для тех полей,
        которых нет в текущей структуре, но которые есть в структуре со значениями по умолчанию.
        Пример:
        A=cStruct(i=1,j=2,k=3)
        B=cStruct(a=10,b=20,c=30,i=333)
        A.GetDefaults(B)
        print A
        Structure: a=10 c=30 b=20 i=1 k=3 j=2 '''
        lsDefaultAttributes = list(vars(oDefaultStruct).keys())
        for sDefaultAttribute in lsDefaultAttributes:
            if not hasattr(self,sDefaultAttribute):
                setattr(self,sDefaultAttribute,getattr(oDefaultStruct,sDefaultAttribute))



    @property
    def fields(self):
        return self._fields

    def get_fields(self):
        return self._fields

    @property
    def fields2values(self):
        dAttributes2Values={}
        for sAttribute in list(vars(self).keys()):
            dAttributes2Values[sAttribute]=getattr(self,sAttribute)
        return dAttributes2Values

    def __setattr__(self, name, value):
            super(Struct, self).__setattr__(name, value)
            if name!='_fields':
                if name not in self._fields:
                    self._fields.append(name)

    def __delattr__(self, name):
            super(Struct, self).__delattr__(name)
            if name!='_fields':
                if name in self._fields:
                    self._fields.remove(name)


''' # Demo #
A=Struct(name='Andrew',Age=23)
A.surname='Osipov'

print A # out: {Age=23, surname=Osipov, name=Andrew}
print A.fields # out: ['Age', 'surname', 'name']
print A.fields2values  # out: {'Age': 23, 'surname': 'Osipov', 'name': 'Andrew'}

D=Struct(name='Undefined',job='Worker')
A.get_defaults(D)
print A # out: {job=Worker, Age=23, surname=Osipov, name=Andrew}

import sys
print sys.getsizeof(A)
'''

if __name__=='__main__':
    '''Demo use:'''

    A=Struct(ccc=3,eee=4)
    A.aaa=1
    A.bbb=2
    print(A._fields)
    print(A.fields)


    S=Struct(a=1,b=2)
    S.c=444
    del S.c
    print(S.c)


