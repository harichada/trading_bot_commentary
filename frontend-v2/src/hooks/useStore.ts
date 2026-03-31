import { useSyncExternalStore } from 'react'
import { getStore, subscribe } from '../lib/store'

export function useStore() {
  return useSyncExternalStore(subscribe, getStore, getStore)
}
