#pragma once

#include <vector>
#include <memory>
//#include <initializer_list>

template <typename T> class VecCRef;

template <typename T>
class VecRef
{
    friend class VecCRef<T>;
public:
    VecRef() : m_p(new std::vector<T>()) {}
    VecRef(size_t size) : m_p(new std::vector<T>(size)) {}
    VecRef(size_t size, const T& t) : m_p(new std::vector<T>(size, t)) {}
    VecRef(const VecRef<T>& rhs) : m_p(rhs.m_p) {}
    VecRef(const std::vector<T>& rhs) : m_p(new std::vector<T>(rhs)) {}
//    VecRef(std::initializer_list<T> list) : m_p(new std::vector<T>(list)) {}

    explicit VecRef(const VecCRef<T>& rhs) // Element-wise copy
        : m_p(new std::vector<T>(rhs.size()))
    {
        for(size_t i = 0; i < rhs.size(); i++)
            operator[](i) = rhs[i];
    }

    typedef typename std::vector<T>::const_iterator const_iterator;

    size_t size() const
    {
        return m_p->size();
    }
    void push_back(const T& t)
    {
        m_p->push_back(t);
    }
    void resize(size_t size)
    {
        m_p->resize(size);
    }
    T& operator[](size_t index) const
    {
        return m_p->operator[](index);
    }
    //const T& operator[](size_t index) const
    //{
    //   return m_p->operator[](index);
    //}
    bool empty() const
    {
        return m_p->empty();
    }
    const_iterator begin() const
    {
        return m_p->begin();
    }
    const_iterator end() const
    {
        return m_p->end();
    }

    operator std::vector<T>&()
    {
        return *m_p;
    }

    void reserve(size_t size)
    {
        m_p->reserve(size);
    }
protected:
    std::shared_ptr<std::vector<T>> m_p;
};

template <typename T>
class VecCRef
{
public:
    VecCRef() : m_p(new std::vector<T>()) {}
    VecCRef(const VecCRef<T>& rhs) : m_p(rhs.m_p) {}
    VecCRef(const VecRef<T>& rhs) : m_p(rhs.m_p) {}
    VecCRef(const std::vector<T>& rhs) : m_p(new std::vector<T>(rhs)) {}

    typedef typename std::vector<T>::const_iterator const_iterator;

    size_t size() const
    {
        return m_p->size();
    }
    const T& operator[](size_t index) const
    {
        return m_p->operator[](index);
    }
    bool empty() const
    {
        return m_p->empty();
    }
    const_iterator begin() const
    {
        return m_p->begin();
    }
    const_iterator end() const
    {
        return m_p->end();
    }
    operator const std::vector<T>&() const
    {
        return *m_p;
    }
    const std::vector<T>& operator *() const
    {
        return *m_p;
    }

protected:
    std::shared_ptr<const std::vector<T>> m_p;
};

