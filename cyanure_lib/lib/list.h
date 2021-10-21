
/* Software SPAMS v2.1 - Copyright 2009-2011 Julien Mairal 
 *
 * This file is part of SPAMS.
 *
 * SPAMS is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SPAMS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SPAMS.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef LIST_H
#define LIST_H

template <typename T> class Element {
   public:
      Element(T el) { element=el; next=NULL; };
      ~Element() {  };
      T element;
      Element<T>* next;
};

template <typename T> class ListIterator {

   public:
      ListIterator() { _current = NULL; };
      ~ListIterator() {  };
      inline void set(Element<T>* elem) { _current = elem; };
      inline T operator*() const { return _current->element; };
      inline bool operator !=(const void* end) const { return _current != end; };
      inline bool operator ==(const void* end) const { return _current == end; };
      inline void operator++() { _current = _current->next; };
      inline Element<T>* current() { return _current; };
      inline T operator->() { return _current->element; };
   private:
      Element<T>* _current;
};

template <typename T> class List {
   public:

      List() { _first=NULL; _last=NULL; _size=0; _iterator = new ListIterator<T>(); };
      ~List() { 
         this->clear();
         delete(_iterator); 
      };
      bool inline empty() const { return _size==0; };
      inline T front() const { 
         return _first->element;
      };
      inline T last() const { 
         return _last->element;
      };
      void inline pop_front() { 
         Element<T>* fr=_first;
         _first=fr->next;
         fr->next=NULL;
         delete(fr);
         --_size;
      };
      void inline push_back(T elem) {
         if (_first) {
            Element<T>* la=_last;
            _last=new Element<T>(elem);
            la->next=_last;
         } else {
            _first=new Element<T>(elem);
            _last=_first;
         }
         ++_size;
      }
      void inline push_front(T elem) {
         Element<T>* fr=_first;
         _first=new Element<T>(elem);
         _first->next=fr;
         if (!_last) _last=_first;
         ++_size;
      }
      void inline clear() {
         ListIterator<T> it = this->begin();
         while (it != this->end()) {
            Element<T>* cur = it.current();
            ++it;
            delete(cur);
         }
         _size=0;
         _first=NULL;
         _last=NULL;
      }
      void inline remove(T elem) {
         if (_first->element == elem) {
            Element<T>* el = _first;
            _first = _first->next;
            delete(el);
         } else {
            Element<T>* old = _first;
            for (ListIterator<T> it = this->begin(); it != this->end(); ++it) {
               if (*it == elem) {
                  Element<T>* el = it.current();
                  old->next=el->next;
                  delete(el);
                  break;
               }
               old=it.current();
            }
         }
      };
      int inline size() const { return _size; };
      inline ListIterator<T>& begin() const { _iterator->set(_first); return *_iterator; };
      inline void* end() const { return NULL; };
      inline void fusion(const List<T>& list) {
         for (ListIterator<T> it = list.begin(); it != list.end(); ++it) {
            this->push_back(*it);
         }
      }
      inline void reverse(List<T>& list) {
         list.clear();
         for (ListIterator<T> it = this->begin(); it != this->end(); ++it) {
            list.push_front(*it);
         }
      }
      inline void copy(List<T>& list) {
         list.clear();
         for (ListIterator<T> it = this->begin(); it != this->end(); ++it) {
            list.push_back(*it);
         }
      }
      void inline print() const {
         cerr << " print list " << endl;
         for (ListIterator<T> it = this->begin(); it != this->end(); ++it) {
            cerr << *it << " ";
         }
         cerr << endl;
      }

   private:

      ListIterator<T>* _iterator;
      Element<T>* _first;
      Element<T>* _last;
      int _size;
};

typedef List<int> list_int;
typedef ListIterator<int> const_iterator_int;


#endif
