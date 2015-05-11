#pragma once

#include <vector>
#include <stdexcept>

#include "Misc/tree.hh"
#include "Array.h"
#include "Rect.h"

/*
  A tree consists of a node structure, a feature test object
  for each internal node, and additional data from training
  stored at each node.
*/
template <class TFeature, class TTrainingData>
struct NodeData
{
    NodeData() {}
    NodeData(const TTrainingData& d) : data(d) {}
    NodeData(const TFeature& f, const TTrainingData& d) : feature(f), data(d) {}
    NodeData(const NodeData& rhs) : feature(rhs.feature), data(rhs.data) {}

    TFeature feature;
    TTrainingData data;
};

template <typename TFeature, typename TTrainingData, typename TAllocator = std::allocator<tree_node_<NodeData<TFeature, TTrainingData> > > >
class DecisionTree : public tree<NodeData<TFeature, TTrainingData>, TAllocator>
{
public:
    typedef tree<NodeData<TFeature, TTrainingData>, TAllocator> TreeBase;
    typedef unsigned int Path;

    class test_time_iterator : public tree<NodeData<TFeature, TTrainingData>, TAllocator>::iterator_base
    {
    public:
        typedef typename tree<NodeData<TFeature, TTrainingData>, TAllocator>::iterator_base Base;

        test_time_iterator() : m_x(-1), m_y(-1), m_prep(nullptr), m_path(0), m_mask(1) {}

        test_time_iterator(const test_time_iterator& rhs) : tree<NodeData<TFeature, TTrainingData>, TAllocator>::iterator_base(rhs), m_x(rhs.m_x), m_y(rhs.m_y), m_prep(rhs.m_prep), m_path(rhs.m_path), m_mask(rhs.m_mask), m_offsets(rhs.m_offsets) {}

        test_time_iterator(typename tree<NodeData<TFeature, TTrainingData>, TAllocator>::tree_node* start, int x, int y, const typename TFeature::PreProcessType& preProcessed, const VecCRef<Vector2D<int> >& offsets)
            : tree<NodeData<TFeature, TTrainingData>, TAllocator>::iterator_base(start), m_x(x), m_y(y), m_prep(&preProcessed), m_offsets(offsets), m_path(0), m_mask(1) {}

        test_time_iterator& operator++()
        {
            if(Base::node->first_child != nullptr)
            {
                bool b = Base::node->data.feature(m_x, m_y, *m_prep, m_offsets);
                Base::node = b ? Base::node->last_child : Base::node->first_child;
#pragma warning(push)
#pragma warning(disable: 4146) // Disable warning about '-' operator applied to unsigned type.
                m_path |= m_mask & (-(Path)b);
#pragma warning(pop)
                m_mask <<= 1;
            }
            else
                Base::node = nullptr;

            return *this;
        }
        operator bool() const
        {
            return Base::node != nullptr;
        }
        bool operator ==(const test_time_iterator& rhs) const
        {
            return Base::node == rhs.node;
        }
        bool operator !=(const test_time_iterator& rhs) const
        {
            return !operator ==(rhs);
        }
        Path path() const
        {
            return m_path;
        }
    protected:
        Path m_path, m_mask;
        int m_x, m_y;
        const typename TFeature::PreProcessType* m_prep;
        VecCRef<Vector2D<int> > m_offsets;
    };

    class path_iterator : public tree<NodeData<TFeature, TTrainingData>, TAllocator>::iterator_base
    {
    public:
        typedef typename tree<NodeData<TFeature, TTrainingData>, TAllocator>::iterator_base Base;
        path_iterator() : m_path(0) {}
        path_iterator(typename tree<NodeData<TFeature, TTrainingData>, TAllocator>::tree_node* start, Path path) : tree<NodeData<TFeature, TTrainingData>, TAllocator>::iterator_base(start), m_path(path) {}
        path_iterator(const path_iterator& rhs) : tree<NodeData<TFeature, TTrainingData>, TAllocator>::iterator_base(rhs), m_path(rhs.m_path) {}

        path_iterator& operator++()
        {
            Base::node = (m_path & 1) ? Base::node->last_child : Base::node->first_child;
            m_path >>= 1;
            return *this;
        }
        bool operator ==(const path_iterator& rhs) const
        {
            return Base::node == rhs.node;
        }
        bool operator !=(const path_iterator& rhs) const
        {
            return !operator ==(rhs);
        }
        operator bool() const
        {
            return Base::node != nullptr;
        }
    protected:
        Path m_path;
    };

    test_time_iterator begin_test(int x, int y, const typename TFeature::PreProcessType& preProcessed, const VecCRef<Vector2D<int> >& offsets) const
    {
        return test_time_iterator(TreeBase::head->next_sibling, x, y, preProcessed, offsets);
    }
    test_time_iterator end_test() const
    {
        return test_time_iterator();
    }
    test_time_iterator goto_leaf(int x, int y, const typename TFeature::PreProcessType& preProcessed, const VecCRef<Vector2D<int> >& offsets) const
    {
        auto i = begin_test(x, y, preProcessed, offsets);

        while(i.node->first_child != nullptr)
            ++i;

        return i;
    }
    Path find_path(int x, int y, const typename TFeature::PreProcessType& preProcessed, const VecCRef<Vector2D<int> >& offsets) const
    {
        return goto_leaf(x, y, preProcessed, offsets).path();
    }
    path_iterator begin_path(Path path) const
    {
        return path_iterator(TreeBase::head->next_sibling, path);
    }
    path_iterator end_path() const
    {
        return path_iterator();
    }
};

template <typename TFeature, typename TTrainingData, typename TAllocator = std::allocator<tree_node_<NodeData<TFeature, TTrainingData> > > >
struct Tree_
{
    typedef DecisionTree<TFeature, TTrainingData, TAllocator> Type;
};

template <typename TFeature, typename TTrainingData, typename TAllocator = std::allocator<tree_node_<NodeData<TFeature, TTrainingData> > > >
class TreeRef
{
public:
    typedef typename Tree_<TFeature, TTrainingData, TAllocator>::Type TBase;

    TreeRef() : m_p(new TBase()) {}
    TreeRef(const TreeRef& rhs) : m_p(rhs.m_p) {}
    TreeRef& operator =(const TreeRef& rhs)
    {
        m_p = rhs.m_p;
        return *this;
    }
    std::shared_ptr<TBase> Ref() const
    {
        return m_p;
    }

    typedef typename TBase::value_type value_type;

    typedef typename TBase::iterator_base iterator_base;
    typedef typename TBase::leaf_iterator leaf_iterator;
    typedef typename TBase::test_time_iterator test_time_iterator;
    typedef typename TBase::pre_order_iterator pre_order_iterator;
    typedef typename TBase::breadth_first_queued_iterator breadth_first_queued_iterator;

    static int depth(const iterator_base& i)
    {
        return TBase::depth(i);
    }
    static unsigned int number_of_children(const iterator_base& i)
    {
        return TBase::number_of_children(i);
    }
    int max_depth() const
    {
        return m_p->max_depth();
    }
    bool empty() const
    {
        return m_p->empty();
    }
    pre_order_iterator set_head(const value_type& value) const
    {
        return m_p->set_head(value);
    }
    pre_order_iterator begin() const
    {
        return m_p->begin();
    }
    pre_order_iterator end() const
    {
        return m_p->end();
    }
    breadth_first_queued_iterator begin_breadth_first() const
    {
        return m_p->begin_breadth_first();
    }
    breadth_first_queued_iterator end_breadth_first() const
    {
        return m_p->end_breadth_first();
    }
    template<typename iter> iter append_child(iter position, const value_type& x) const
    {
        return m_p->append_child(position, x);
    }
    template<typename iter> iter append_child(iter position) const
    {
        return m_p->append_child(position);
    }
    leaf_iterator begin_leaf() const
    {
        return m_p->begin_leaf();
    }
    leaf_iterator end_leaf() const
    {
        return m_p->end_leaf();
    }
    test_time_iterator goto_leaf(int x, int y, const typename TFeature::PreProcessType& preProcessed, const VecCRef<Vector2D<int> >& offsets) const
    {
        return m_p->goto_leaf(x, y, preProcessed, offsets);
    }
    size_t size() const
    {
        return m_p->size();
    }

private:
    std::shared_ptr<TBase> m_p;
};

template <typename TFeature, typename TTrainingData, typename TAllocator = std::allocator<tree_node_<NodeData<TFeature, TTrainingData> > > >
class TreeCRef
{
public:
    typedef typename Tree_<TFeature, TTrainingData, TAllocator>::Type TBase;

    TreeCRef() : m_p(new TBase()) {}
    TreeCRef(const TreeRef<TFeature, TTrainingData, TAllocator>& rhs) : m_p(rhs.Ref()) {}
    TreeCRef(const TreeCRef& rhs) : m_p(rhs.m_p) {}
    TreeCRef& operator =(const TreeCRef& rhs)
    {
        m_p = rhs.m_p;
        return *this;
    }

    typedef unsigned int Path;
    typedef typename TBase::value_type value_type;

    typedef typename TBase::iterator_base iterator_base;
    typedef typename TBase::test_time_iterator test_time_iterator;
    typedef typename TBase::leaf_iterator leaf_iterator;
    typedef typename TBase::pre_order_iterator pre_order_iterator;
    typedef typename TBase::breadth_first_queued_iterator breadth_first_queued_iterator;

    static int depth(const iterator_base& i)
    {
        return TBase::depth(i);
    }
    static unsigned int number_of_children(const iterator_base& i)
    {
        return TBase::number_of_children(i);
    }
    bool empty() const
    {
        return m_p->empty();
    }
    int max_depth() const
    {
        return m_p->max_depth();
    }
    pre_order_iterator begin() const
    {
        return m_p->begin();
    }
    pre_order_iterator end() const
    {
        return m_p->end();
    }
    leaf_iterator begin_leaf() const
    {
        return m_p->begin_leaf();
    }
    leaf_iterator end_leaf() const
    {
        return m_p->end_leaf();
    }
    breadth_first_queued_iterator begin_breadth_first() const
    {
        return m_p->begin_breadth_first();
    }
    breadth_first_queued_iterator end_breadth_first() const
    {
        return m_p->end_breadth_first();
    }
    test_time_iterator begin_test(int x, int y, const typename TFeature::PreProcessType& preProcessed, const VecCRef<Vector2D<int> >& offsets) const
    {
        return m_p->begin_test(x, y, preProcessed, offsets);
    }
    test_time_iterator end_test() const
    {
        return m_p->end_test();
    }
    test_time_iterator goto_leaf(int x, int y, const typename TFeature::PreProcessType& preProcessed, const VecCRef<Vector2D<int> >& offsets) const
    {
        return m_p->goto_leaf(x, y, preProcessed, offsets);
    }
    Path find_path(int x, int y, const typename TFeature::PreProcessType& preProcessed, const VecCRef<Vector2D<int> >& offsets) const
    {
        return m_p->find_path(x, y, preProcessed, offsets);
    }
    size_t size() const
    {
        return m_p->size();
    }

    template <typename TOp>
    void WalkTreeData(int x, int y, const typename TFeature::PreProcessType& prep, const VecCRef<Vector2D<int> >& offsets, TOp& op) const
    {
        for(auto i = begin_test(x, y, prep, offsets); i; ++i)
            op(i->data);

        //if (empty())
        //   return;
        //auto node = begin().node;
        //op(node->data.data);
        //while (node->first_child != nullptr)
        //{
        //   bool b = node->data.feature(x, y, prep);
        //   node = b ? node->last_child : node->first_child;
        //   op(node->data.data);
        //}
    }

#if 0
    template <typename TOp>
    void WalkPathData(Path path, TOp& op) const
    {
        for(auto i = begin_path(path); i; ++i)
            op(i->data);

        //auto node = begin().node;
        //op(node->data.data);
        //while (node->first_child != nullptr)
        //{
        //   node = (path & 1) ? node->last_child : node->first_child;
        //   path >>= 1;
        //   op(node->data.data);
        //}
    }
#endif

    const TTrainingData& GetLeafData(int x, int y, const typename TFeature::PreProcessType& prep, const VecCRef<Vector2D<int> >& offsets) const
    {
        return goto_leaf(x, y, prep, offsets)->data;
        //if (empty())
        //   throw std::exception("Empty tree");
        //auto node = begin().node;
        //while (node->first_child != nullptr)
        //{
        //   bool b = node->data.feature(x, y, prep);
        //   node = b ? node->last_child : node->first_child;
        //}
        //return node->data.data;
    }

private:
    std::shared_ptr<const TBase> m_p;
};

class TreeTable
{
public:
    typedef int Path;
    static const Path BadPath = -1;
    typedef unsigned int NodeId;
    virtual ~TreeTable() {}
    virtual unsigned int GetSizeofEntry() const = 0;
    virtual unsigned int GetSizeofTrainingData() const = 0;
    virtual unsigned int GetSizeofFeature() const = 0;
    virtual unsigned int GetEntryCount() const = 0;
    virtual unsigned char* GetFirstData() = 0;
};

template <typename TFeature, typename TTrainingData>
class TreeTableT : public TreeTable
{
public:
    TreeTableT() {}
    virtual ~TreeTableT() {}

    template <typename T2, typename TOp>
    void Fill(const TreeCRef<TFeature, T2>& root, TOp op)
    {
        // Pre-allocate table
        m_entries.reserve(root.size());
        int nThis = 0, nNext = 1;

        for(auto i = root.begin_breadth_first(); i != root.end_breadth_first(); ++i)
        {
            bool bLeaf = i.number_of_children() == 0;
            Entry entry;
            entry.feature = i->feature;
            entry.data = op(i);
            entry.entrySkip = bLeaf ? -1 : nNext - nThis;
            m_entries.push_back(entry);
            nNext += i.number_of_children();
            nThis++;
        }
    }

    template <typename T2, typename TOp>
    TreeRef<TFeature, T2> BuildTree(TOp& op) const
    {
        TreeRef<TFeature, T2> tree;

        if(m_entries.empty())
            return tree;

        tree.set_head(NodeData<TFeature, T2>());
        auto i = tree.begin_breadth_first();

        for(unsigned int entry = 0; entry < m_entries.size(); entry++, ++i)
        {
            i->feature = m_entries[entry].feature;
            i->data = op(m_entries[entry].data);

            if(m_entries[entry].entrySkip >= 0)
            {
                tree.append_child(i);
                tree.append_child(i);
            }
        }
    }

    template <typename TOutput>
    void WalkTree(int x, int y, const typename TFeature::PreProcessType& preProcessed, const VecCRef<Vector2D<int> >& offsets, TOutput& output) const
    {
        // For each pixel independently,
        // walk the tree, and tell the output object which
        // nodes are being visited.
        const Entry* pEntry = &m_entries[0];

        while(pEntry->entrySkip >= 0)
        {
            output(pEntry->data);
            bool b = pEntry->feature(x, y, preProcessed, offsets);
            pEntry += pEntry->entrySkip + b;
        }

        output(pEntry->data);
    }

    Path FindPath(int x, int y, const typename TFeature::PreProcessType& preProcessed, const VecCRef<Vector2D<int> >& offsets) const
    {
        Path path = 0, mask = 1;
        const Entry* pEntry = &m_entries[0];

        while(pEntry->entrySkip >= 0)
        {
            bool b = pEntry->feature(x, y, preProcessed, offsets);
            path |= mask & (-(Path)b);
            pEntry += pEntry->entrySkip + b;
            mask <<= 1;
        }

        return path;
    }

    // Returns the index of the leaf node, using a breadth-first numbering
    TTrainingData GetLeafData(int x, int y, const typename TFeature::PreProcessType& preProcessed, const VecCRef<Vector2D<int> >& offsets) const
    {
        const Entry* pEntry = &m_entries[0];

        while(pEntry->entrySkip >= 0)
        {
            bool b = pEntry->feature(x, y, preProcessed, offsets);
            pEntry += pEntry->entrySkip + b;
        }

        return pEntry->data;
    }

    template <typename TOutput>
    void WalkPath(Path path, TOutput& output) const
    {
        int entry = 0;
        const Entry* pEntry = &m_entries[0];

        while(pEntry->entrySkip >= 0)
        {
            output(entry, pEntry->data);
            int skip = pEntry->entrySkip + (path & 1);
            entry += skip;
            pEntry += skip;
            path >>= 1;
        }

        output(entry, pEntry->data);
    }

    template <typename TOutput>
    void WalkPath(Path path, TOutput& output)
    {
        int entry = 0;
        Entry* pEntry = &m_entries[0];

        while(pEntry->entrySkip >= 0)
        {
            output(entry, pEntry->data);
            int skip = pEntry->entrySkip + (path & 1);
            entry += skip;
            pEntry += skip;
            path >>= 1;
        }

        output(entry, pEntry->data);
    }

    unsigned int GetSizeofEntry() const
    {
        return sizeof(Entry);
    }
    unsigned int GetSizeofTrainingData() const
    {
        return sizeof(TTrainingData);
    }
    unsigned int GetSizeofFeature() const
    {
        return sizeof(TFeature);
    }
    unsigned int GetEntryCount() const
    {
        return (unsigned int)m_entries.size();
    }
    unsigned char* GetFirstData()
    {
        return (unsigned char*)&m_entries[0].data;
    }

    void ZeroData()
    {
        unsigned int size = sizeof(TTrainingData);

        for(auto it = m_entries.begin(); it != m_entries.end(); ++it)
        {
            memset(&(it->data), 0, size);
        }
    }

protected:
    struct Entry
    {
        Entry() {}
        Entry(const Entry& rhs) : feature(rhs.feature), data(rhs.data), entrySkip(rhs.entrySkip) {}

        TFeature feature;
        TTrainingData data;
        int entrySkip; ///< Number of entries to skip forward for a false test, -1 designates leaf node
    };
    std::vector<Entry> m_entries;
};

template <unsigned char variableCount, unsigned char labelCount> struct power
{
    static const unsigned int raise;
};
template <unsigned char labelCount> struct power<1, labelCount>
{
    static const unsigned int raise = labelCount;
};
template <unsigned char labelCount> struct power<2, labelCount>
{
    static const unsigned int raise = labelCount * labelCount;
};
// t-shaib
template <unsigned char labelCount> struct power<3, labelCount>
{
    static const unsigned int raise = labelCount * labelCount * labelCount;
};
template <unsigned char labelCount> struct power<4, labelCount>
{
    static const unsigned int raise = labelCount * labelCount * labelCount * labelCount;
};
template <unsigned char labelCount> struct power<5, labelCount>
{
    static const unsigned int raise = labelCount * labelCount * labelCount * labelCount * labelCount;
};

template <unsigned char labelCount> struct power<6, labelCount>
{
    static const unsigned int raise = labelCount * labelCount * labelCount *
                                      labelCount * labelCount * labelCount;
};
template <unsigned char labelCount> struct power<7, labelCount>
{
    static const unsigned int raise = labelCount * labelCount * labelCount * labelCount *
                                      labelCount * labelCount * labelCount ;
};
template <unsigned char labelCount> struct power<8, labelCount>
{
    static const unsigned int raise = labelCount * labelCount * labelCount * labelCount *
                                      labelCount * labelCount * labelCount * labelCount;
};
#if 0
// XXX: -sn removed, this creates an integral constant overflow
template <unsigned char labelCount> struct power<9, labelCount>
{
    static const unsigned int raise = labelCount * labelCount * labelCount *
                                      labelCount * labelCount * labelCount *
                                      labelCount * labelCount * labelCount;
};
#endif
