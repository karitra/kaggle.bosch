//
// author: Alex Karev
// date: 2016.09.22
// Simple scanner for Kaggle Bosch contest
//
#include <algorithm>
#include <bitset>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <forward_list>
#include <iostream>
#include <limits>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <boost/range/adaptors.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/range/algorithm_ext.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/utility/string_ref.hpp>

// #include <boost/phoenix.hpp>
// using namespace boost::phoenix::arg_names;

#include "Eigen/Dense"

using namespace Eigen;
using namespace std;

namespace {

const char SPLIT_DELIM = ',';

const int FEATURE_BITSET_SIZE = 16 * 1024;
const int BITS_OUT = 32;
const int VALS_OUT = 6;
const int APPROX_RECORDS = 10 * 1000; // TODO: make it command line parameters
const int MVG_MULTIPLIER = 2;

const double SPECIAL_VALUE = 1e-1;

enum TResp {
  OK_RESP,
  BROKEN_RESP,
  ALL_RESP
};


typedef std::pair<int, int> TColsPair;
typedef std::pair<bitset<FEATURE_BITSET_SIZE>, int> THashCount;
typedef std::unordered_set<int> TMissedSet;


template<size_t Z>
std::ostream &
operator<<(std::ostream &os, const std::bitset<Z> &bs) {
  const auto out_len = std::min(bs.size(), static_cast<size_t>(BITS_OUT));
  for(int i = 0; i < out_len; ++i ) { os << (bs[i] ? 'T' : 'F'); }

  return os;
}

} // namespace

template<class T, int BitSize = FEATURE_BITSET_SIZE>
class CRecord {
public:
  typedef std::bitset<BitSize> TDefaultBS;

  CRecord(const int i, const bool r, const std::bitset<BitSize> &b, const std::vector<T> &init) :
    Id(i), Response(r), bs(b), payload(init) {}

  friend std::ostream &operator<<(std::ostream &os, const CRecord<T> &r) {
    os << "Id: " << r.Id << " Re: " << r.Response << ' ';
    os << r.bs << ' ';

    const auto out_len = std::min(r.payload.size(), static_cast<size_t>(VALS_OUT));
    for(int i = 0; i < out_len; ++i) {
      os << r.payload[i] << ' ';
    }

    return os;
  }

  RowVectorXf MakeVec(const TMissedSet &remIdx) const {

    const auto srcVec = MakeVec();

    assert(srcVec.size() > remIdx.size());

    RowVectorXf rv(srcVec.size() - remIdx.size());

    int it = 0;
    for(int i = 0; i < srcVec.size(); ++i) {
      if (remIdx.count(i) == 0) {
        rv(it++) = srcVec(i);
      }
    }

    return rv;
  }

  RowVectorXf MakeVec() const {

    RowVectorXf v(bs.count());
    int it = 0;

    for(int i = 0; i <= payload.size(); ++i) {
      if (bs[i]) {
        // cout << "val: " << payload[i] << '\n';
        v(it++) = payload[i];
      }
    }

    // cout << "vec(" << bs.count() << "): " << v << '\n';
    return v;
  }

  int Id;
  bool Response;

  const TDefaultBS &GetBSRef() const { return bs; }
  const std::vector<T> &GetPayloadRef() const { return payload; }

private:
  std::bitset<BitSize> bs; // 512 bytes (for 4k bits)
  std::vector<T> payload;
};

template<class T>
const CRecord<T> scan(const std::string &ln, const TColsPair &cols) {

  int string_start = 0;
  std::vector<T> data;
  std::bitset<FEATURE_BITSET_SIZE> non_nan_flags;
  data.reserve(FEATURE_BITSET_SIZE);

  const auto str = ln.data();
  int i = 0;
  int col = 0;

  int id = 0;
  bool r = 0;

  for(const auto &ch : ln) {

    i++;

    switch(ch) {
      // case '\n':
      case SPLIT_DELIM:
        {
          const auto len = i - 1 - string_start;

          if (len == 0) { // nan
            data.emplace_back(std::numeric_limits<T>::quiet_NaN());
          } else {

            const auto sr = boost::string_ref(str + string_start, len);

            if ( (col == cols.first || col == cols.second) == false) {
              // cerr << "val => " << sr << '\n';

              if (sr[0] != 'T') {
                try {
                  data.emplace_back( stof(sr.data()) );
                  non_nan_flags.set(data.size() - 1);
                } catch(const std::invalid_argument &ia) {
                    cerr << "invalid argument to stof: [" << sr << "]\n";
                }
              } else {
                data.emplace_back(std::numeric_limits<T>::quiet_NaN());
              }

            } else if (col == cols.first) { // id
               // cerr << "id => " << sr << '\n';
               id = stoi(sr.data());
            } else if (col == cols.second) { // response
                r = static_cast<bool>( stoi(sr.data()) );
                // cerr << "resp => " << sr << " r = " << r << '\n';
            }
          }

          ++col;
          string_start = i;
        }
        break;
    }
  }

  ++i;
  const auto len = i - 1 - string_start;
  if (len == 0) { // nan
    data.push_back( std::numeric_limits<T>::quiet_NaN() );
  } else {
    // cout << "string_start => " << string_start << " total len => " << ln.size() << " len => " << len <<  '\n';
    const auto sr = boost::string_ref(str + string_start, len);
    if (sr[0] != 'T') {
      // cerr << "** val => [" << &str[string_start] << "]";
      data.push_back( stof(sr.data()) );
      non_nan_flags.set(data.size());
    } else {
      data.push_back( std::numeric_limits<T>::quiet_NaN() );
    }
  }

  return CRecord<T>(id, r, non_nan_flags, data);
}

std::vector<string> get_names(const std::string &ln) {
  std::vector<string> names;
  boost::algorithm::split(names, ln, boost::algorithm::is_any_of(","));
  return names;
}



template<class N>
struct UnitType {

    typedef std::forward_list<CRecord<N> *> TPlainRecsList;


    UnitType(const size_t fc = 0) : fieldsCount(fc) {}

    size_t size() const { return okCount + brokenCount; }
    size_t GetOkCount() const { return okCount; }
    size_t GetBrokenCount() const { return brokenCount; }

    void add(const CRecord<N> *r) {
      items.push_back(r);

      if (r->Response == 1) {
        ++brokenCount;
      } else {
        ++okCount;
      }
    }

    forward_list<const CRecord<N> *> GetBrokenRecs() const { return GetByResponseRecs(true); }

    forward_list<int> GetBrokenIds() const { return GetByResponseIds(false); }
    forward_list<int> GetOkIds() const { return GetByResponseIds(true); }

    forward_list<const CRecord<N> *> GetByResponseRecs(const bool resp) const {
      forward_list<const CRecord<N> *> result;

      std::copy_if(begin(items), end(items), std::front_inserter(result),
        [&resp] (const CRecord<N> *el) {
            return el->Response == resp;
        });

      return result;
    }


    forward_list<int> GetByResponseIds(const bool resp) const {
      forward_list<int> result;

      boost::transform(
        items | boost::adaptors::filtered( [&resp] (const CRecord<N> *el) {
        el->Response == resp; }),
        result,
        [] (const CRecord<N> *el) { el->Id; });

      return result;
    }

    size_t GetFieldsCount() const { return fieldsCount; }
    void SetFieldsCount(const size_t &fCnt) { fieldsCount = fCnt; }

    std::pair<shared_ptr<MatrixXf>, TMissedSet> MakeMatrix() const {
      MatrixXf m(okCount, fieldsCount);

      // cout << "matrix dim: " << m.rows() << ' ' << m.cols() << '\n';

      int row = 0;
      for(const auto &el : items) {
        assert(el != nullptr);

        if (el->Response == 1) continue;

        // cout << "  vec.dim: " << v.size() << '\n';
        m.row(row++) = el->MakeVec();
      }

      // const auto z = VectorXf::Zero(m.cols());
      TMissedSet removeIdx;
      removeIdx.reserve(m.cols());

      //
      // Removing zero columns
      //
      for(int c = 0; c < m.cols(); ++c ) {
        // cout << "z.size = " << z.size() << " m.col = " << m.col(c).size() << endl;
        // cout << " m.col = " << m.col(c).size() << endl;
        if (m.col(c).isConstant(0.0)) removeIdx.insert(c);
      }

      shared_ptr<MatrixXf> tm = make_shared<MatrixXf>(m);

#if 1
      if (removeIdx.empty() == false) {

        tm = std::make_shared<MatrixXf>(m.rows(), m.cols() - removeIdx.size());

        int cl = 0;
        for(int i = 0; i < m.cols(); ++i) {
          if (removeIdx.count(i) == 0) {
            tm->col(cl++) = m.col(i);
          }
        }
      }

      cout << "    mat. size: " << m.rows() << ' ' << m.cols() << endl;
      cout << "new mat. size: " << tm->rows() << ' ' << tm->cols() << endl;
#endif

      assert(tm);

      return std::make_pair(tm, removeIdx);
    }

    int okCount = 0;
    int brokenCount = 0;

    size_t fieldsCount = 0;

    std::list<const CRecord<N> *> items;
};

typedef std::unordered_map<bitset<FEATURE_BITSET_SIZE>, std::shared_ptr<UnitType<double>> > TBitMapping;
typedef std::bitset<FEATURE_BITSET_SIZE> TBitSet;

template<class T>
struct DataTable {

  typedef std::forward_list<CRecord<T> *> TPlainRecsList;

  DataTable(int recCount = APPROX_RECORDS) {
    recs.reserve(recCount);
    bm.reserve(recCount);
    idMap.reserve(recCount);
  }

  void add(const CRecord<T> &&r) {
    recs.push_back(r);

    auto it = bm.find(r.GetBSRef());
    const auto &newElemPtr = &recs.back();
    if (it != end(bm) ) {
      it->second->add( newElemPtr );
    } else {
      bm[r.GetBSRef()] = std::make_shared< UnitType<T> >( r.GetBSRef().count() );
      bm[r.GetBSRef()]->add( newElemPtr );
    }

    idMap.insert( make_pair(r.Id, newElemPtr ) );

    if (r.Response == 1) {
      brokenList.push_front(&recs.back());
      brokenCount++;
    } else {
      okList.push_front(&recs.back());
      okCount++;
    }
  }

  void clear() {
      recs.clear();
      bm.clear();

      brokenList.clear();
      okList.clear();

      unitsHistList.clear();
      okUnitsHist.clear();
      brokenUnitsHist.clear();

      okCount = brokenCount = 0;

      idMap.clear();
  }

  const CRecord<T> *ById(int id) const {

    const auto it = idMap.find(id);

    if (it != end(idMap) ) {
      return it->second;
    } else {
      return nullptr;
    }
  }

  std::forward_list<CRecord<T> *> ByIds(const std::forward_list<int> &ids) {
    std::forward_list<CRecord<T> *> res;
    for(const auto &id : ids) {
      const auto p = ById(id);
      if (p != nullptr) {
        res.push_front(p);
      }
    }

    return res;
  }

  std::shared_ptr< UnitType<T> > ByGroup(const TBitSet &flags) {
      auto it = bm.find(flags);
      if (it == end(bm)) {
          return nullptr;
      } else {
          return it->second;
      }
  }

  MatrixXf MakeGroupMatrix(const TBitSet &g) {
    auto grp = ByGroup(g);
    if (grp) {
      return grp->MakeMatrix();
    } else {
      return MatrixXf(0, 0);
    }
  }

  std::list<THashCount> &RebuildHist() {
    return rebuildHist(unitsHistList, TResp::ALL_RESP);
  }

  std::list<THashCount> &RebuildOkHist() {
    return rebuildHist(okUnitsHist, TResp::OK_RESP);
  }

  std::list<THashCount> &RebuildBrokenHist() {
    return rebuildHist(brokenUnitsHist, TResp::BROKEN_RESP);
  }

  std::list<THashCount> &rebuildHist(std::list<THashCount> &hist, const TResp &t) {

    hist.clear();

    switch (t) {
      case OK_RESP:

        boost::transform(bm, std::back_inserter(hist),
          [] (const TBitMapping::value_type &v) {
          return make_pair(v.first, v.second->GetOkCount() );
        });

        break;
      case BROKEN_RESP:
        boost::transform(bm, std::back_inserter(hist),
          [] (const TBitMapping::value_type &v) {
            return make_pair(v.first, v.second->GetBrokenCount() );
          });

        break;
      case ALL_RESP:
        boost::transform(bm, std::back_inserter(hist),
          [] (const TBitMapping::value_type &v) {
            return make_pair(v.first, v.second->size() );
          });

        break;
    }

    hist.remove_if( [] (const THashCount &a) { return a.second == 0 ; });

    hist.sort( [] (const THashCount &a, const THashCount &b) {
      return a.second > b.second;
    } );

    return hist;
  }

  std::forward_list<int> GetBrokenIds() {
    return GetIdsList(okList);
  }

  std::forward_list<int> GetOkIds() const {
    return GetIdsList(brokenList);
  }

  std::forward_list<int> GetIdsList(const TPlainRecsList &src) const {
    std::forward_list<int> res;

    boost::transform( src, res, [] (const typename TPlainRecsList::reference r) {
      return r.Id;
    });

    return res;
  }

  size_t size() const { return recs.size(); }

  const TPlainRecsList &GetBroken() const { return brokenList; }
  const TPlainRecsList &GetOk() const { return okList; }

  TBitMapping bm;
  std::vector<CRecord<T>> recs;
  std::list<THashCount> unitsHistList;
  std::list<THashCount> brokenUnitsHist;
  std::list<THashCount> okUnitsHist;

  std::unordered_map<int, CRecord<T> *> idMap; // id => record mapping

  int brokenCount = 0;
  int okCount = 0;

  TPlainRecsList brokenList;
  TPlainRecsList okList;
};


void dumpHist(ostream &os, const std::list<THashCount> &hist, int lim=6) {
  auto i = 0;

  os << "hist size: " << hist.size() << '\n';
  for(const auto &v : hist) {
    os << v.second << ' ' << v.first << '\n';
    if (++i >= lim) {
      break;
    }
  }

  os << '\n';
}

template<class M, class V>
bool isInvertable(const M &m, const V &x, const int mul = 1) {
  assert(mul > 0);
  return m.rows() > x.size() * mul;
}

template<class M, class V>
bool isAllowableMVG(const M &m, const V &x) {
  return isInvertable(m,x, MVG_MULTIPLIER);
}


//
// TODO: data normalization
//
template<class M, class V>
double mvgInTime(const M &m, const V &x) {

  const auto mu = m.colwise().mean();
  // const auto m_delta = m.rowwise() - mu;
  const auto m_delta = m.rowwise() - mu;
  const M sigma = (m_delta.adjoint() * m_delta) / m.rows();

#if 0
  cerr << "x size: " << x.rows() << ' ' << x.cols() << '\n';
  cerr << "mu size: " << mu.rows() << ' ' << mu.cols() << '\n';
#endif

  const auto x_delta = x - mu;

#if 0
  cout << "x_delta size: " << x_delta.rows() << ' ' << x_delta.cols() << '\n';
  cout << "sigma size: "   <<   sigma.rows() << ' ' <<   sigma.cols() << '\n';
#endif

  const V csum = sigma.colwise().sum();
  if (csum.isConstant(0.0)) {
    cout << "BOOM sigma == 0\n";
  } else {
    for (int i = 0; i < csum.size(); ++i) {
      if (abs(csum(i) - 0.0) < 0.0000001) {
        cout << "BOOM @ " << i << " val " << csum(i) << "\n";
      }
    }
  }

  cout << "det: " << sigma.determinant() << " sigma:\n" << sigma.topLeftCorner(10, 8) << endl;

  return 1.0 /
    sqrt( pow(2.0 * M_PI, m.rows()) * sigma.norm() ) *
      ( x_delta * sigma.inverse() * x_delta.adjoint() ).array().exp()(0,0);
}

template<class M, class V>
double
checkOutlier(const M &m, const V &x, const double mult = 1) {

  const auto N = m.rows() > 1 ?
    m.rows() - 1 :
    1;

  const V mu = m.colwise().mean();

#if 1
  cerr << "mu rows: " << mu.rows() << " cols: " << mu.cols() << '\n';
  cerr << " m rows: " <<  m.rows() << " cols: " <<  m.cols() << '\n';
  cerr << " x rows: " <<  x.rows() << " cols: " <<  x.cols() << '\n';
#endif

  // const M v1 = m.rowwise() - mu;
  // cerr << "here1\n";
  V var = (m.rowwise() - mu).array().square().colwise().sum() / int(N);

#if 1
  cerr << "var.hasNaNs: " << std::boolalpha << var.hasNaN() << '\n';
  // cerr << "var.isFinite: " << var.isFinite() << '\n';
  cerr << "var.hasZero: " << std::boolalpha << (var.array() == 0.0).count() << '\n';
  cerr << " var rows: " << var.rows() << " cols: " << var.cols() << '\n';
  cerr << " mu: "       << mu.head(6) << '\n';
  cerr << " var: "      << var.head(6) << '\n';
#endif


  // Clear variance from zeros and nans (shouldn't be here)
  for(int i = 0; i < var.size(); ++i) {
    if (std::isnan(var(i)) || var(i) == 0) {
      var(i) = SPECIAL_VALUE;
    }
  }

  const V p =
    1 /
    (2 * M_PI * var).array().sqrt()
    * ( -(x - mu).array().square().array()
        / (2 * var).array() ).array().exp();

  cerr << "p size: " << p.rows() << ' ' << p.cols() << endl;
  cerr << "p: " << p.head(5) << " ... " << p.tail(5) << '\n';
  // cerr << "m.col[29] \n" << m.col(28) << '\n';
  cerr << "mu[29] " << mu[28] << '\n';
  cerr << "var[29] " << var[28] << '\n';
  cerr << "p[29] " << p[28] << '\n';
  cerr << "p.prod10: " << p.head(10).prod() << '\n';
  cerr << "p.prod15: " << p.head(15).prod() << '\n';
  cerr << "p.prod20: " << p.head(20).prod() << '\n';
  cerr << "p.prod25: " << p.head(25).prod() << '\n';
  cerr << "p.prod27: " << p.head(27).prod() << '\n';
  cerr << "p.prod28: " << p.head(28).prod() << '\n';
  cerr << "p.prod29: " << p.head(29).prod() << '\n';

  cout << "\np.vec:\n" << p << '\n';
  return (mult*p).prod();
}


int main(int argc, char *argv[])
{
  using namespace boost::adaptors;
  using namespace boost::algorithm;

  std::string ln;

  if (!getline(cin, ln)) {
    cout << "Failed to read first line\n";
    return EXIT_FAILURE;
  }

  auto names = get_names(ln);
  assert(names.empty() == false);

  auto idCol = -1;
  auto respCol = -1;

  const auto indexedNames = names | indexed(0);
  #if 0
      for(auto it = boost::begin(indexedNames); it != boost::end(indexedNames); ++it) {
        cout << it.index() << ' ' << *it << '\n';
      }
  #endif


  const auto itId = boost::find( indexedNames, "Id");
  const auto itResp = boost::find( indexedNames, "Response");

  if (itId != boost::end(indexedNames)) {
      idCol = itId.index();
  }

  if (itResp != boost::end(indexedNames)) {
      respCol = itResp.index();
  }

  assert(idCol != -1);
  assert(respCol != -1);

  DataTable<double> dt;
  while(getline(cin, ln)) {
    dt.add( scan<double>(ln, make_pair(idCol, respCol) ) );
  }

  cout << "units: " << dt.size() << '\n';

  dumpHist(cout, dt.RebuildHist() );
  dumpHist(cout, dt.RebuildOkHist() );

  auto brHist = dt.RebuildBrokenHist();
  dumpHist(cout, brHist);

  for (const auto &brokenGrp : brHist) {
    const auto sptrGrp = dt.ByGroup(brokenGrp.first);

    cout << brokenGrp.second << ' ';

    if (sptrGrp) {
      cout << sptrGrp->GetOkCount() << ' ' << brokenGrp.first << '\n';

      const auto mRes = sptrGrp->MakeMatrix();

      const auto m  = mRes.first;
      const auto rmIdx = mRes.second;

      if (!m) {
        cout << " all zero columns\n";
        continue;
      }

      if (sptrGrp->GetOkCount() == 0) {
        cout << "  empty set\n";
        continue;
      }

      auto faultUnits = sptrGrp->GetBrokenRecs();

      auto mu = m->colwise().mean();

      cout << "mat: \n" << m->middleCols(25, 7) << '\n';

      for(const auto &fu : faultUnits) {
        cout << "  id: " << fu->Id << ' ';

        auto brVec = fu->MakeVec(rmIdx);

        if (isAllowableMVG(*m, brVec)) {
          auto d = mvgInTime(*m, brVec);
          cout << " d: " << d << ' ';
        } else {
          int a[] = {1,2,3,5,10,20,60};
          cout << " ";
          for(const auto &i : a) {
            auto p = checkOutlier(*m, brVec, i);
            cout << "p" << i << ": " << p << ' ';
          }
        }

        auto mx = (m->rowwise() - brVec).rowwise().norm().maxCoeff();
        cout << " max dist: " << mx << ' ';

        auto mu_dist = (mu - brVec).norm();
        cout << " dist from mean: " << mu_dist << '\n';
      }

    } else {
      cout << "  na \n";
    }

  }

  return EXIT_SUCCESS;
}
